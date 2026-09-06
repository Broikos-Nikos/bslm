"""Train BSLM from random init. Joint loss: intent CE + slot CE."""
import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from .model import BSLM
from .tokenizer import BPETokenizer

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CKPT = ROOT / "checkpoints"


def read(split):
    with (DATA / f"{split}.jsonl").open(encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def build_labels(rows):
    intents = sorted({r["intent"] for r in rows})
    slots = sorted({t[2:] for r in rows for t in r["tags"] if t != "O"})
    tags = ["O"] + [f"{p}-{s}" for s in slots for p in ("B", "I")]
    return intents, slots, tags


class DS(Dataset):
    def __init__(self, rows, tok, intent2i, tag2i, max_len, augment=0.0):
        self.rows, self.tok = rows, tok
        self.i2, self.t2, self.max_len = intent2i, tag2i, max_len
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        ids, cases, widx = self.tok.encode(r["words"], self.max_len)
        if self.augment:
            # token dropout, stops the model leaning on one keyword
            ids = [1 if (j and random.random() < self.augment) else t
                   for j, t in enumerate(ids)]
        labels = []
        for w in widx:
            labels.append(-100 if w < 0 else self.t2.get(r["tags"][w], 0))
        return (torch.tensor(ids), torch.tensor(cases), torch.tensor(labels),
                torch.tensor(self.i2[r["intent"]]))


def collate(batch):
    n = max(len(b[0]) for b in batch)
    ids = torch.zeros(len(batch), n, dtype=torch.long)
    cases = torch.zeros(len(batch), n, dtype=torch.long)
    labels = torch.full((len(batch), n), -100, dtype=torch.long)
    intents = torch.stack([b[3] for b in batch])
    for i, (a, c, l, _) in enumerate(batch):
        ids[i, :len(a)] = a
        cases[i, :len(c)] = c
        labels[i, :len(l)] = l
    return ids, cases, labels, intents


def spans_from_bio(tags):
    out, cur = [], None
    for i, t in enumerate(tags):
        if t.startswith("B-"):
            if cur:
                out.append(cur)
            cur = [t[2:], i, i]
        elif t.startswith("I-") and cur and cur[0] == t[2:]:
            cur[2] = i
        else:
            if cur:
                out.append(cur)
            cur = None
    if cur:
        out.append(cur)
    return {tuple(s) for s in out}


@torch.no_grad()
def evaluate(model, loader, tags, device):
    model.eval()
    ok_i = tot = 0
    tp = fp = fn = 0
    joint = 0
    for ids, cases, labels, intents in loader:
        ids, cases = ids.to(device), cases.to(device)
        li, ls = model(ids, cases)
        pi = li.argmax(-1).cpu()
        ps = ls.argmax(-1).cpu()
        ok_i += (pi == intents).sum().item()
        tot += len(intents)
        for b in range(len(intents)):
            keep = labels[b] != -100
            gold = spans_from_bio([tags[t] for t in labels[b][keep].tolist()])
            pred = spans_from_bio([tags[t] for t in ps[b][keep].tolist()])
            tp += len(gold & pred)
            fp += len(pred - gold)
            fn += len(gold - pred)
            if pi[b] == intents[b] and gold == pred:
                joint += 1
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    model.train()
    return ok_i / tot, f1, joint / tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--d_ff", type=int, default=768)
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--vocab", type=int, default=4000)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--augment", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=1337, help="fixed so retrains are comparable")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    CKPT.mkdir(exist_ok=True)
    train_rows, val_rows, test_rows = read("train"), read("val"), read("test")
    intents, slots, tags = build_labels(train_rows)
    i2 = {v: i for i, v in enumerate(intents)}
    t2 = {v: i for i, v in enumerate(tags)}
    print(f"train {len(train_rows)}  val {len(val_rows)}  test {len(test_rows)}")
    print(f"{len(intents)} intents, {len(slots)} slots, {len(tags)} bio tags")

    tok_path = CKPT / "tokenizer.json"
    if tok_path.exists():
        tok = BPETokenizer.load(tok_path)
    else:
        print("training bpe...")
        tok = BPETokenizer.train([r["text"] for r in train_rows], args.vocab)
        tok.save(tok_path)
    print(f"vocab {len(tok)}")

    dl = lambda rows, sh, aug=0.0: DataLoader(
        DS(rows, tok, i2, t2, args.max_len, aug), batch_size=args.batch, shuffle=sh,
        collate_fn=collate, num_workers=0)
    tl = dl(train_rows, True, args.augment)
    vl, sl = dl(val_rows, False), dl(test_rows, False)

    model = BSLM(len(tok), len(intents), len(tags), args.d_model, args.layers,
                 args.heads, args.d_ff, args.max_len, args.dropout).to(device)
    print(f"{model.n_params()/1e6:.2f}M parameters on {device}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01,
                            betas=(0.9, 0.98))
    steps = args.epochs * len(tl)
    warm = max(50, int(steps * 0.05))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / warm if s < warm else
        0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    ce_i = nn.CrossEntropyLoss(label_smoothing=0.05)
    ce_s = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.02)

    best, t0 = 0.0, time.time()
    for ep in range(1, args.epochs + 1):
        run = 0.0
        for ids, cases, labels, intent in tl:
            ids, cases = ids.to(device), cases.to(device)
            labels, intent = labels.to(device), intent.to(device)
            li, ls = model(ids, cases)
            loss = ce_i(li, intent) + ce_s(ls.reshape(-1, ls.size(-1)), labels.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run += loss.item()
        acc, f1, joint = evaluate(model, vl, tags, device)
        print(f"ep {ep:2d}  loss {run/len(tl):.4f}  val intent {acc:.4f}  "
              f"slot f1 {f1:.4f}  joint {joint:.4f}  {time.time()-t0:.0f}s")
        if joint >= best:
            best = joint
            tmp = CKPT / "bslm.tmp"
            torch.save({"model": model.state_dict(), "cfg": model.cfg,
                        "intents": intents, "tags": tags, "slots": slots,
                        "max_len": args.max_len}, tmp)
            os.replace(tmp, CKPT / "bslm.pt")            # atomic, never a half written file

    ck = torch.load(CKPT / "bslm.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    acc, f1, joint = evaluate(model, sl, tags, device)
    print(f"\nHELD OUT TEST (unseen phrasings)  intent {acc:.4f}  slot f1 {f1:.4f}  joint {joint:.4f}")
    print(f"saved {CKPT/'bslm.pt'}")


if __name__ == "__main__":
    main()
