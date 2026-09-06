"""Fine tune a pretrained checkpoint on the trajectories (stage 5 and 6).

    python -m pretrain.sft --name 72m-agent --init 72m-10b --epochs 3

Loss only where the model speaks (Plan, Act, Judge, Ask, Deliver and the end
of text after Deliver); the header, the user's words and every Result block
are context. Trajectories are packed whole into 1024 token windows, never
split, so the model always sees the header of the episode it is learning.
Same optimiser as pretraining (Muon and AdamW) at a tenth of the rates,
short warmup, linear decay over the second half.
"""
import argparse
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .model import Config, Llama
from .train import Muon, lr_scale, maybe_compile, has_triton

ROOT = Path(__file__).resolve().parent.parent
AGENT = ROOT / "corpus" / "agent"
RUNS = ROOT / "pretrain" / "runs"
EOT_ID = None


def pack(ids, mask, seq, eot):
    """Split the token stream at end of text, pack whole documents into windows
    of seq+1 tokens (x is [:-1], y is [1:]); pad with eot and mask 0."""
    cuts = np.flatnonzero(ids == eot)
    docs, start = [], 0
    for c in cuts:
        docs.append((ids[start:c + 1], mask[start:c + 1]))
        start = c + 1
    docs = [d for d in docs if len(d[0]) <= seq + 1]
    xs, ms, cur_i, cur_m = [], [], [], []
    for di, dm in docs:
        if len(cur_i) + len(di) > seq + 1:
            xs.append(cur_i)
            ms.append(cur_m)
            cur_i, cur_m = [], []
        cur_i = cur_i + di.tolist()
        cur_m = cur_m + dm.tolist()
    if cur_i:
        xs.append(cur_i)
        ms.append(cur_m)
    X = np.full((len(xs), seq + 1), eot, dtype=np.int64)
    M = np.zeros((len(xs), seq + 1), dtype=np.float32)
    for k, (a, b) in enumerate(zip(xs, ms)):
        X[k, :len(a)] = a
        M[k, :len(b)] = b
    return torch.from_numpy(X), torch.from_numpy(M)


def masked_loss(logits, y, m):
    l = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.reshape(-1), reduction="none")
    return (l * m.reshape(-1)).sum() / m.sum().clamp(min=1)


@torch.no_grad()
def greedy(model, tok, prompt, max_new=120, stop=("\nResult:", "\nUser:")):
    ids = tok.encode(prompt).ids
    x = torch.tensor([ids], device="cuda")
    out = []
    for _ in range(max_new):
        logits = model(x[:, -model.cfg.seq:])[0] if isinstance(model(x[:, -model.cfg.seq:]), tuple) else model(x[:, -model.cfg.seq:])
        nxt = int(logits[0, -1].argmax())
        if nxt == EOT_ID:
            break
        out.append(nxt)
        x = torch.cat([x, torch.tensor([[nxt]], device="cuda")], 1)
        text = tok.decode(out)
        if any(s in text for s in stop):
            break
    return tok.decode(out)


def main():
    global EOT_ID
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--init", required=True, help="run name of the pretrained checkpoint")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--micro", type=int, default=8)
    ap.add_argument("--batch_tokens", type=int, default=131072)
    ap.add_argument("--lr_muon", type=float, default=0.004)
    ap.add_argument("--lr_adam", type=float, default=4e-4)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--decay_frac", type=float, default=0.5)
    ap.add_argument("--eval_every", type=int, default=50)
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

    init = torch.load(RUNS / args.init / "ckpt.pt", map_location=device, weights_only=False)
    cfg = Config(**{k: v for k, v in init["cfg"].items() if k in Config.__init__.__code__.co_varnames})
    cfg.seq = args.seq
    model = Llama(cfg).to(device)
    model.load_state_dict(init["model"])
    say(f"init from {args.init} step {init.get('step')}  {model.n_params()/1e6:.1f}M params")

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(ROOT / "pretrain" / "tokenizer.json"))
    EOT_ID = tok.token_to_id("<|endoftext|>")
    Xtr, Mtr = pack(np.load(AGENT / "train.npy"), np.load(AGENT / "train_mask.npy"), args.seq, EOT_ID)
    Xva, Mva = pack(np.load(AGENT / "val.npy"), np.load(AGENT / "val_mask.npy"), args.seq, EOT_ID)
    say(f"train {Xtr.size(0)} windows ({Xtr.numel()/1e6:.1f}M tokens, {100*Mtr.mean():.0f}% trained)  val {Xva.size(0)} windows")

    hidden = [p for n, p in model.named_parameters() if p.ndim == 2 and "tok_emb" not in n]
    other = [p for n, p in model.named_parameters() if not (p.ndim == 2 and "tok_emb" not in n)]
    opt_m = Muon(hidden, lr=args.lr_muon, momentum=0.95, weight_decay=args.wd)
    opt_a = torch.optim.AdamW(other, lr=args.lr_adam, betas=(0.9, 0.95), weight_decay=0.0, fused=True)
    accum = max(1, args.batch_tokens // (args.micro * args.seq))
    steps_per_epoch = max(1, Xtr.size(0) // (args.micro * accum))
    total = int(args.epochs * steps_per_epoch)
    say(f"steps {total}  ({steps_per_epoch} per epoch, micro {args.micro} x accum {accum})")
    fwd = model if args.no_compile else maybe_compile(model)
    say(f"compile: {'on' if (not args.no_compile and has_triton()) else 'off'}")

    prompts = ["Today is Monday 2026-09-07, 10:00. Home: Athens. Facts: rides a scooter.\nUser: who directed the film Inception\n",
               "Today is Monday 2026-09-07, 10:00. Home: Athens. Facts: rides a scooter.\nUser: set a timer for 12 minutes\n",
               "Today is Monday 2026-09-07, 10:00. Home: Athens. Facts: rides a scooter.\nUser: play hotel california by the eagles\n"]

    def evaluate():
        model.eval()
        losses = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, min(Xva.size(0), 64), args.micro):
                xb = Xva[i:i + args.micro].to(device)
                mb = Mva[i:i + args.micro].to(device)
                logits = fwd(xb[:, :-1])
                logits = logits[0] if isinstance(logits, tuple) else logits
                losses.append(masked_loss(logits, xb[:, 1:], mb[:, 1:]).item())
            outs = [p.split("User: ")[1].strip() + " | " + greedy(model, tok, p).replace("\n", " || ") for p in prompts]
        model.train()
        return sum(losses) / max(1, len(losses)), outs

    def save():
        tmp = run / "ckpt.tmp"
        torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "args": vars(args), "step": step,
                    "init": args.init}, tmp)
        os.replace(tmp, run / "ckpt.pt")

    g = torch.Generator().manual_seed(1337)
    step, t0, last = 0, time.time(), time.time()
    model.train()
    while step < total:
        s = lr_scale(step, total, args.warmup, args.decay_frac)
        for grp in opt_m.param_groups:
            grp["lr"] = args.lr_muon * s
        for grp in opt_a.param_groups:
            grp["lr"] = args.lr_adam * s
        loss_acc = 0.0
        for _ in range(accum):
            idx = torch.randint(0, Xtr.size(0), (args.micro,), generator=g)
            xb, mb = Xtr[idx].to(device, non_blocking=True), Mtr[idx].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = fwd(xb[:, :-1])
                logits = logits[0] if isinstance(logits, tuple) else logits
                loss = masked_loss(logits, xb[:, 1:], mb[:, 1:])
            (loss / accum).backward()
            loss_acc += loss.item() / accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt_m.step()
        opt_a.step()
        opt_m.zero_grad(set_to_none=True)
        opt_a.zero_grad(set_to_none=True)
        step += 1
        if step % 10 == 0 or step == 1:
            now = time.time()
            say(f"step {step:4d}/{total}  loss {loss_acc:.4f}  lr x{s:.3f}  {(now - last):.1f}s/10  "
                f"epoch {step / steps_per_epoch:.2f}  mem {torch.cuda.max_memory_allocated()/1e9:.1f}GB")
            last = now
        if step % args.eval_every == 0 or step == total:
            vl, outs = evaluate()
            say(f"eval step {step}  val loss {vl:.4f}  ppl {math.exp(vl):.2f}")
            with (run / "samples.txt").open("a", encoding="utf-8") as f:
                f.write(f"\n=== step {step}  val {vl:.4f} ===\n" + "\n".join(outs) + "\n")
            for o in outs:
                say("  " + o[:200])
            save()
    say(f"done {step} steps in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
