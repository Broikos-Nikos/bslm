"""Pretrain from random init on one RTX 4060. Resumable, interruptible.

    python -m pretrain.train --name 56m-fineweb --size 56m --tokens 1e9
    python -m pretrain.train --name 56m-fineweb --resume            (continue)
    python -m pretrain.train --name 56m-fineweb --resume --tokens 2e9  (extend)

Muon on the hidden matrices, AdamW on embeddings and norms, bf16, compiled,
warmup / stable / decay schedule so a run can be extended without restart.
Checkpoint every --ckpt_every steps (about hourly) to pretrain/runs/<name>/.
"""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .model import SIZES, Llama

ROOT = Path(__file__).resolve().parent.parent
TOK = ROOT / "corpus" / "tokens"
RUNS = ROOT / "pretrain" / "runs"


def has_triton():
    try:
        import triton  # noqa: F401
        return True
    except Exception:
        return False


def maybe_compile(fn):
    return torch.compile(fn) if has_triton() and os.environ.get("BSLM_NO_COMPILE") != "1" else fn


# ---------------------------------------------------------------- muon
@maybe_compile
def zeropower_via_newtonschulz5(G, steps=5):
    """Orthogonalise G (Newton-Schulz iteration, bf16). From Keller Jordan's Muon."""
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov,
                                      ns_steps=ns_steps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                st = self.state[p]
                if "buf" not in st:
                    st["buf"] = torch.zeros_like(grad)
                buf = st["buf"]
                buf.mul_(g["momentum"]).add_(grad)
                upd = grad.add(buf, alpha=g["momentum"]) if g["nesterov"] else buf
                upd = zeropower_via_newtonschulz5(upd, g["ns_steps"])
                upd = upd * max(1, p.size(0) / p.size(1)) ** 0.5
                if g["weight_decay"]:
                    p.mul_(1 - g["lr"] * g["weight_decay"])
                p.add_(upd.type_as(p), alpha=-g["lr"])


# ---------------------------------------------------------------- data
class Stream:
    """Contiguous windows over the token shards, position is resumable."""

    def __init__(self, split, micro, seq):
        pat = "train_*.npy" if split == "train" else "val.npy"
        self.files = sorted(TOK.glob(pat))
        assert self.files, f"no shards for {split} in {TOK}"
        self.micro, self.seq = micro, seq
        self.shard_i, self.pos, self.epoch = 0, 0, 0
        self._load()

    def _order(self):
        # epoch 0 reads the shards in file order; every later epoch reads them
        # in a fresh seeded permutation, so multi epoch runs do not replay the
        # identical token sequence (the seed makes resume reproducible)
        order = list(range(len(self.files)))
        if self.epoch > 0:
            random.Random(1000 + self.epoch).shuffle(order)
        return order

    def _load(self):
        self.data = np.load(self.files[self._order()[self.shard_i]], mmap_mode="r")

    def state(self):
        return {"shard_i": self.shard_i, "pos": self.pos, "epoch": self.epoch}

    def restore(self, s):
        self.shard_i, self.pos, self.epoch = s["shard_i"], s["pos"], s["epoch"]
        self._load()

    def next(self, device):
        need = self.micro * self.seq + 1
        if self.pos + need > len(self.data):
            self.shard_i = (self.shard_i + 1) % len(self.files)
            if self.shard_i == 0:
                self.epoch += 1
            self.pos = 0
            self._load()
        buf = torch.from_numpy(self.data[self.pos:self.pos + need].astype(np.int64))
        self.pos += self.micro * self.seq
        x = buf[:-1].view(self.micro, self.seq)
        y = buf[1:].view(self.micro, self.seq)
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


# ---------------------------------------------------------------- schedule
def lr_scale(step, total, warmup, decay_frac):
    if step < warmup:
        return (step + 1) / warmup
    start = int(total * (1 - decay_frac))
    if step < start:
        return 1.0
    return max(0.0, (total - step) / max(1, total - start))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--size", default="56m", choices=list(SIZES))
    ap.add_argument("--tokens", type=float, default=1e9)
    ap.add_argument("--batch_tokens", type=int, default=524288)
    ap.add_argument("--micro", type=int, default=16)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--lr_muon", type=float, default=0.02)
    ap.add_argument("--lr_adam", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=150)
    ap.add_argument("--decay_frac", type=float, default=0.2)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--ckpt_every", type=int, default=150)
    ap.add_argument("--max_steps", type=int, default=0, help="stop early (sanity runs)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no_compile", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = "cuda"
    run = RUNS / args.name
    run.mkdir(parents=True, exist_ok=True)
    log = (run / "log.txt").open("a", encoding="utf-8")

    def say(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    cfg = SIZES[args.size]
    cfg.seq = args.seq
    model = Llama(cfg).to(device)
    say(f"model {args.size}: {model.n_params()/1e6:.1f}M params  {cfg}")

    hidden = [p for n, p in model.named_parameters() if p.ndim == 2 and "tok_emb" not in n]
    other = [p for n, p in model.named_parameters() if not (p.ndim == 2 and "tok_emb" not in n)]
    opt_m = Muon(hidden, lr=args.lr_muon, momentum=0.95, weight_decay=args.wd)
    opt_a = torch.optim.AdamW(other, lr=args.lr_adam, betas=(0.9, 0.95), weight_decay=0.0, fused=True)

    accum = args.batch_tokens // (args.micro * args.seq)
    total_steps = int(args.tokens // args.batch_tokens)
    train = Stream("train", args.micro, args.seq)
    val = Stream("val", args.micro, args.seq)
    step = 0
    ck = run / "ckpt.pt"
    if args.resume and ck.exists():
        st = torch.load(ck, map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        opt_m.load_state_dict(st["opt_m"])
        opt_a.load_state_dict(st["opt_a"])
        train.restore(st["train"])
        step = st["step"]
        say(f"resumed at step {step} ({step*args.batch_tokens/1e9:.2f}B tokens)")
    say(f"steps {total_steps}  batch {args.batch_tokens} tokens  micro {args.micro}x{args.seq}  accum {accum}")

    fwd = model if args.no_compile else maybe_compile(model)
    say(f"compile: {'on' if (not args.no_compile and has_triton()) else 'off (no triton)'}")
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(ROOT / "pretrain" / "tokenizer.json"))
    prompts = ["The capital of France is", "To set a timer on your phone, you",
               "Water boils at", "Once upon a time"]

    def evaluate():
        model.eval()
        losses = []
        val.restore({"shard_i": 0, "pos": 0, "epoch": 0})
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(20):
                x, y = val.next(device)
                _, l = fwd(x, y)
                losses.append(l.item())
            outs = []
            for p in prompts:
                ids = torch.tensor([tok.encode(p).ids], device=device)
                o = model.generate(ids, max_new=40, temperature=0.8, top_k=40)
                outs.append(p + " |" + tok.decode(o[0, ids.size(1):].tolist()).replace("\n", " "))
        model.train()
        return sum(losses) / len(losses), outs

    def save():
        # write to a temp file and rename, so a crash mid save never leaves a
        # corrupt ckpt.pt behind for the restart loop to choke on
        tmp = ck.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "opt_m": opt_m.state_dict(),
                    "opt_a": opt_a.state_dict(), "train": train.state(), "step": step,
                    "cfg": cfg.__dict__, "args": vars(args)}, tmp)
        os.replace(tmp, ck)

    model.train()
    t0 = time.time()
    tokens_done = step * args.batch_tokens
    last = time.time()
    while step < total_steps:
        s = lr_scale(step, total_steps, args.warmup, args.decay_frac)
        for g in opt_m.param_groups:
            g["lr"] = args.lr_muon * s
        for g in opt_a.param_groups:
            g["lr"] = args.lr_adam * s
        loss_acc = 0.0
        for _ in range(accum):
            x, y = train.next(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = fwd(x, y)
            (loss / accum).backward()
            loss_acc += loss.item() / accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt_m.step()
        opt_a.step()
        opt_m.zero_grad(set_to_none=True)
        opt_a.zero_grad(set_to_none=True)
        step += 1
        tokens_done += args.batch_tokens
        if step % 10 == 0 or step == 1:
            now = time.time()
            tps = 10 * args.batch_tokens / (now - last) if step > 1 else args.batch_tokens / (now - last)
            last = now
            eta = (total_steps - step) * args.batch_tokens / max(tps, 1) / 3600
            say(f"step {step:5d}/{total_steps}  loss {loss_acc:.4f}  lr x{s:.3f}  "
                f"{tps/1e3:6.1f}k tok/s  {tokens_done/1e9:.3f}B tokens  epoch {train.epoch}  "
                f"eta {eta:.1f}h  mem {torch.cuda.max_memory_allocated()/1e9:.1f}GB")
        if step % args.eval_every == 0:
            vl, outs = evaluate()
            say(f"eval step {step}  val loss {vl:.4f}  ppl {math.exp(vl):.1f}")
            with (run / "samples.txt").open("a", encoding="utf-8") as f:
                f.write(f"\n=== step {step}  val {vl:.4f} ===\n" + "\n".join(outs) + "\n")
        if step % args.ckpt_every == 0:
            save()
            say(f"checkpoint saved at step {step}")
        if args.max_steps and step >= args.max_steps:
            say("max_steps reached")
            break
    save()
    vl, outs = evaluate()
    say(f"final step {step}  val loss {vl:.4f}  ppl {math.exp(vl):.1f}  {(time.time()-t0)/3600:.2f}h")
    for o in outs:
        say("  " + o)


if __name__ == "__main__":
    main()
