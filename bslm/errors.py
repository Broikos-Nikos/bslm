"""Per intent breakdown on the adversarial test split, worst first."""
import json
from collections import Counter, defaultdict
from pathlib import Path

from .infer import Parser

ROOT = Path(__file__).resolve().parent.parent


def main():
    rows = [json.loads(l) for l in
            (ROOT / "data" / "test.jsonl").open(encoding="utf-8")]
    p = Parser()
    per = defaultdict(lambda: [0, 0])
    confusion = defaultdict(Counter)
    for r in rows:
        got = p.parse(r["text"])["intent"]
        per[r["intent"]][1] += 1
        if got == r["intent"]:
            per[r["intent"]][0] += 1
        else:
            confusion[r["intent"]][got] += 1
    ranked = sorted(per.items(), key=lambda kv: kv[1][0] / kv[1][1])
    print(f"{'intent':24s} {'acc':>7s}  n     confused with")
    for intent, (ok, n) in ranked:
        worst = ", ".join(f"{k} {v}" for k, v in confusion[intent].most_common(3))
        print(f"{intent:24s} {ok/n:7.1%}  {n:<5d} {worst}")
    tot_ok = sum(v[0] for v in per.values())
    tot = sum(v[1] for v in per.values())
    print(f"\noverall {tot_ok/tot:.1%} on {tot} adversarial rows")


if __name__ == "__main__":
    main()
