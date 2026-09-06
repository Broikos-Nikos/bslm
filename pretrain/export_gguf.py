"""Export a checkpoint to GGUF so llama.cpp (and a phone) can run it.

    python -m pretrain.export_gguf --name 56m-fineweb            -> models/56m-fineweb-f16.gguf
    tools\\llama\\llama-quantize.exe models\\56m-fineweb-f16.gguf models\\56m-fineweb-q8.gguf Q8_0

The architecture is Llama shaped, so this only renames tensors and permutes
Q and K into llama.cpp's rotary layout, the same permutation the official HF
converter applies. Tied embeddings: no output.weight is written, llama.cpp
falls back to token_embd.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from gguf import GGUFWriter, TokenType

from .model import Config, Llama

ROOT = Path(__file__).resolve().parent.parent


def permute(w, n_head):
    """rotate-half layout -> interleaved pairs, per row block of head_dim."""
    return (w.reshape(n_head, 2, w.shape[0] // n_head // 2, *w.shape[1:])
             .swapaxes(1, 2).reshape(w.shape))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ck = torch.load(ROOT / "pretrain" / "runs" / args.name / "ckpt.pt",
                    map_location="cpu", weights_only=False)
    cfg = Config(**ck["cfg"])
    model = Llama(cfg)
    model.load_state_dict(ck["model"])
    sd = {k: v.float().numpy() for k, v in model.state_dict().items()}

    out = Path(args.out) if args.out else ROOT / "models" / f"{args.name}-f16.gguf"
    out.parent.mkdir(exist_ok=True)
    w = GGUFWriter(str(out), "llama")
    w.add_name(args.name)
    w.add_context_length(cfg.seq)
    w.add_embedding_length(cfg.dim)
    w.add_block_count(cfg.n_layers)
    w.add_feed_forward_length(cfg.ffn)
    w.add_head_count(cfg.n_heads)
    w.add_head_count_kv(cfg.n_kv_heads)
    w.add_rope_dimension_count(cfg.head_dim)
    w.add_rope_freq_base(cfg.rope_theta)
    w.add_layer_norm_rms_eps(cfg.norm_eps)
    w.add_file_type(1)                                   # F16

    tok = json.loads((ROOT / "pretrain" / "tokenizer.json").read_text(encoding="utf-8"))
    vocab = tok["model"]["vocab"]
    tokens = [None] * len(vocab)
    for t, i in vocab.items():
        tokens[i] = t
    specials = {a["content"] for a in tok.get("added_tokens", [])}
    types = [TokenType.CONTROL if t in specials else TokenType.NORMAL for t in tokens]
    merges = [" ".join(m) if isinstance(m, list) else m for m in tok["model"]["merges"]]
    eot = vocab["<|endoftext|>"]
    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre("gpt-2")
    w.add_token_list(tokens)
    w.add_token_types(types)
    w.add_token_merges(merges)
    w.add_bos_token_id(eot)
    w.add_eos_token_id(eot)
    w.add_add_bos_token(False)

    def put(name, arr):
        # llama.cpp wants 1D tensors (norm weights) in f32; matrices go f16
        w.add_tensor(name, arr.astype(np.float32 if arr.ndim == 1 else np.float16))

    put("token_embd.weight", sd["tok_emb.weight"])
    put("output_norm.weight", sd["norm.weight"])
    for i in range(cfg.n_layers):
        p = f"blocks.{i}."
        put(f"blk.{i}.attn_norm.weight", sd[p + "attn_norm.weight"])
        put(f"blk.{i}.attn_q.weight", permute(sd[p + "attn.wq.weight"], cfg.n_heads))
        put(f"blk.{i}.attn_k.weight", permute(sd[p + "attn.wk.weight"], cfg.n_kv_heads))
        put(f"blk.{i}.attn_v.weight", sd[p + "attn.wv.weight"])
        put(f"blk.{i}.attn_output.weight", sd[p + "attn.wo.weight"])
        put(f"blk.{i}.ffn_norm.weight", sd[p + "ffn_norm.weight"])
        put(f"blk.{i}.ffn_gate.weight", sd[p + "mlp.w1.weight"])
        put(f"blk.{i}.ffn_up.weight", sd[p + "mlp.w3.weight"])
        put(f"blk.{i}.ffn_down.weight", sd[p + "mlp.w2.weight"])
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out}  ({out.stat().st_size/1e6:.1f} MB, step {ck['step']})")


if __name__ == "__main__":
    main()
