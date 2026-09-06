"""Generate from a checkpoint.

    python -m pretrain.sample --name 56m-fineweb "The capital of France is"
"""
import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer

from .model import Config, Llama

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--max_new", type=int, default=60)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("prompts", nargs="+")
    args = ap.parse_args()

    ck = torch.load(ROOT / "pretrain" / "runs" / args.name / "ckpt.pt",
                    map_location="cuda", weights_only=False)
    model = Llama(Config(**ck["cfg"])).cuda().eval()
    model.load_state_dict(ck["model"])
    tok = Tokenizer.from_file(str(ROOT / "pretrain" / "tokenizer.json"))
    print(f"{args.name}: step {ck['step']}, {model.n_params()/1e6:.1f}M params\n")
    for p in args.prompts:
        ids = torch.tensor([tok.encode(p).ids], device="cuda")
        for _ in range(args.n):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model.generate(ids, args.max_new, args.temperature, args.top_k)
            print(f"> {p}\n  {tok.decode(out[0, ids.size(1):].tolist())}\n")


if __name__ == "__main__":
    main()
