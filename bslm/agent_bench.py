"""Stage 7: the agent benchmark. Held out tasks with known truth, the model in
the loop with real tools, four loop metrics with bars.

    python -m bslm.agent_bench --limit 120        writes AGENT_BENCHMARK.md

Tasks come from corpus/agent/test.jsonl, written by the trajectory generator
and never trained on (held out facts and songs, fresh local tasks). Every
task is run through bslm.agent.Agent against llama-server on the CPU with
the real tools, and judged by rule:

- fact: the delivered sentence contains the known answer; when the answer
  was not findable, an honest "could not" counts, an invented one does not
- song: the played item is the song (library or the matching YouTube title)
- local, compound: the expected action ran with the expected arguments
- umbrella: the advice matches the real forecast (rain or dry)
- other: answered without a tool call and without inventing

Loop metrics (from OWN_MODEL.md stage 7): recovery rate (correct after a
failed first attempt), false delivery (says done but wrong), wasted steps
(the same action repeated unchanged), honest give up (says so when it
cannot be done).
"""
import argparse
import json
import os
import random
import re
import tempfile
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from pretrain.agent_tools import Env, parse_act, rain_hours
from pretrain.trajectories import found_in, norm
from .agent import Agent

ROOT = Path(__file__).resolve().parent.parent
TEST = ROOT / "corpus" / "agent" / "test.jsonl"
FROZEN = ROOT / "corpus" / "agent" / "test_frozen.jsonl"
FROZEN_FAMILIES = ("fact", "song", "umbrella", "compound", "other")   # fixed since round six; local and followup are template made
OUT = ROOT / "AGENT_BENCHMARK.md"
FAIL_WORDS = ("no results", "no match", "there is no", "not found", "failed", "nothing", "could not", "unknown")
GIVEUP_WORDS = ("could not", "cannot", "can't", "did not find", "couldn't", "not find", "no result")


def same_act(expected, got):
    a, b = parse_act(expected), parse_act(got)
    if not a or not b or a[0] != b[0]:
        return False
    fa = [norm(json.dumps(x)) if isinstance(x, (list, tuple)) else norm(x) for x in a[1]]
    fb = [norm(json.dumps(x)) if isinstance(x, (list, tuple)) else norm(x) for x in b[1]]
    # trailing empty arguments do not matter
    while fa and not fa[-1]:
        fa.pop()
    while fb and not fb[-1]:
        fb.pop()
    return fa == fb


def judge(task, r, env, raw=False):
    """-> (correct, honest_expected, honest_said, invented); raw judges the
    model's own delivery, before the runtime's check."""
    fam = task["family"]
    ans = r.get("raw_answer", r["answer"]) if raw else r["answer"]
    acts = [a for a, _ in r["trace"]]
    said_giveup = any(w in ans.lower() for w in GIVEUP_WORDS)
    if fam == "followup":
        if task.get("kind") == "fact":
            fam = "fact"
        else:
            ok = all(any(same_act(e, a) for a in acts) for e in task["acts"])
            return ok, False, said_giveup, False
    if fam == "local" and task.get("act") is None:
        # a tool that is not registered: no tool call, and say so
        ok = not any(a.startswith("tool(") for a in acts) and ("no tool" in ans.lower() or "no tools" in ans.lower())
        return ok, True, ok, not ok
    if fam == "fact":
        if task["answer"] is None:
            return said_giveup, True, said_giveup, not said_giveup
        ok = r["kind"] == "deliver" and found_in(task["answer"], ans) or any(found_in(x, ans) for x in task.get("aliases", []))
        return ok, False, said_giveup, (not ok and not said_giveup and r["kind"] == "deliver")
    if fam == "song":
        played = env.playing or ""
        if task["expected"] == "none":
            return said_giveup, True, said_giveup, not said_giveup
        ok = norm(task["title"]) in norm(played) or (task["expected"] == "youtube" and task.get("video") and norm(task["video"]) == norm(played))
        return ok, False, said_giveup, (not ok and not said_giveup)
    if fam == "local":
        if task.get("ask"):
            ok = r["kind"] == "ask"
            return ok, False, False, False
        ok = any(same_act(task["act"], a) for a in acts)
        return ok, False, said_giveup, False
    if fam == "compound":
        ok = all(any(same_act(e, a) for a in acts) for e in task["acts"])
        return ok, False, said_giveup, False
    if fam == "umbrella":
        low = ans.lower()
        # the truth is the forecast as it stands now, not as it stood when the
        # task was written: "tomorrow" has moved on since then
        expected = task["expected"]
        if task.get("place") and task.get("hour") is not None:
            rh = rain_hours(task["place"], task["day"], env.now)
            if rh is None:
                expected = "no_forecast"
            else:
                rainy = rh[0]
                expected = "rain" if any(f"{h:02d}:00" in rainy for h in (task["hour"] - 2, task["hour"] - 1, task["hour"])) else "dry"
        if expected == "rain":
            ok = any(w in low for w in ("rain", "umbrella", "car", "jacket")) and "no rain" not in low
        elif expected == "dry":
            ok = "no rain" in low or "go as you are" in low
        else:
            ok = said_giveup
        return ok, expected == "no_forecast", said_giveup, False
    if fam == "other":
        ok = not acts and r["kind"] == "deliver" and len(ans) > 3
        return ok, False, said_giveup, False
    return False, False, False, False


def run_task(agent, task, rng):
    """Fresh state per task; setup actions replayed for local tasks."""
    f = tempfile.NamedTemporaryFile(prefix="bslm_bench_", suffix=".json", delete=False)
    f.close()
    os.remove(f.name)
    library = None
    if task["family"] == "song":
        library = [f"{task['title']} - {task['artist']}"] if task.get("in_library") else []
        library += ["Bohemian Rhapsody - Queen", "Billie Jean - Michael Jackson", "Yesterday - The Beatles"]
    env = Env(f.name, now=datetime.now().replace(second=0, microsecond=0), library=library, tools=task.get("tools") or [])
    for a in task.get("setup") or []:
        env.act(a)
    agent.env = env
    agent.tools = [tuple(t) for t in (task.get("tools") or [])]
    agent.memory = [tuple(e) for e in (task.get("earlier") or [])]
    agent.last_time = None
    t0 = time.time()
    r = agent.run(task["user"])
    if r["kind"] == "ask" and task.get("ask"):
        r2 = agent.reply(task["ask"][1])
        r2["trace"] = r["trace"] + r2["trace"]
        r2["kind_first"] = "ask"
        r2["wasted"] += r["wasted"]
        r = r2
        r["kind"] = "ask"       # the first move was the ask, which is what the task tests
    r["seconds"] = time.time() - t0
    try:
        os.remove(f.name)
    except OSError:
        pass
    return r, env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=120, help="tasks per family")
    ap.add_argument("--model", default=None)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    fresh = [json.loads(l) for l in TEST.open(encoding="utf-8")]
    frozen = [json.loads(l) for l in FROZEN.open(encoding="utf-8")] if FROZEN.exists() else []
    tasks = [t for t in frozen if t["family"] in FROZEN_FAMILIES] + [t for t in fresh if t["family"] not in FROZEN_FAMILIES]
    if not frozen:
        tasks = fresh
    by_fam = defaultdict(list)
    for t in tasks:
        by_fam[t["family"]].append(t)
    chosen = []
    for fam, ts in sorted(by_fam.items()):
        rng.shuffle(ts)
        chosen += ts[:args.limit]
    agent = Agent(env=Env(ROOT / "data" / "bench_state.json"), model=args.model)
    per = defaultdict(lambda: [0, 0])          # correct, n  (with the runtime check)
    per_raw = defaultdict(int)                 # correct, the model alone
    per_rel = defaultdict(lambda: [0, 0])      # facts by relation
    loop = {"recover_ok": 0, "recover_n": 0, "false": 0, "deliver_n": 0, "wasted": 0, "steps": 0,
            "giveup_ok": 0, "giveup_n": 0}
    misses, t0 = [], time.time()
    try:
        for i, task in enumerate(chosen):
            try:
                r, env = run_task(agent, task, rng)
            except Exception as e:      # noqa: BLE001
                print(f"task failed: {task['user'][:60]!r}: {str(e)[:80]}", flush=True)
                r, env = {"kind": "fail", "answer": "", "trace": [], "wasted": 0}, Env(ROOT / "data" / "bench_state.json")
            ok, honest_expected, honest_said, invented = judge(task, r, env)
            ok_raw = judge(task, r, env, raw=True)[0]
            per[task["family"]][1] += 1
            per[task["family"]][0] += ok
            per_raw[task["family"]] += ok_raw
            if task["family"] == "fact":
                per_rel[task["rel"]][1] += 1
                per_rel[task["rel"]][0] += ok
            failed_first = any(any(w in res.lower() for w in FAIL_WORDS) for _, res in r["trace"][:1])
            if failed_first:
                loop["recover_n"] += 1
                loop["recover_ok"] += ok
            if r["kind"] == "deliver":
                loop["deliver_n"] += 1
                loop["false"] += invented
            loop["wasted"] += r["wasted"]
            loop["steps"] += max(1, len(r["trace"]))
            if honest_expected:
                loop["giveup_n"] += 1
                loop["giveup_ok"] += honest_said
            if not ok:
                misses.append((task["family"], task["user"], r["answer"][:140], [a for a, _ in r["trace"]][:4]))
            if (i + 1) % 20 == 0:
                print(f"{i + 1}/{len(chosen)}  {time.time() - t0:.0f}s  " +
                      "  ".join(f"{k} {100 * v[0] / v[1]:.0f}%" for k, v in sorted(per.items())), flush=True)
    finally:
        agent.stop()
    rows = ["| family | n | model alone | with the delivery check | bar |", "|---|---|---|---|---|"]
    fails = 0
    for fam, (c, n) in sorted(per.items()):
        pct = 100 * c / n
        fails += pct < 80
        rows.append(f"| {fam} | {n} | {100 * per_raw[fam] / n:.1f}% | {pct:.1f}% | {'pass' if pct >= 80 else '**FAIL**'} |")
    tot_c, tot_n = sum(v[0] for v in per.values()), sum(v[1] for v in per.values())
    rec = 100 * loop["recover_ok"] / max(1, loop["recover_n"])
    fal = 100 * loop["false"] / max(1, loop["deliver_n"])
    was = 100 * loop["wasted"] / max(1, loop["steps"])
    giv = 100 * loop["giveup_ok"] / max(1, loop["giveup_n"])
    md = f"""# Agent benchmark

Held out tasks (`corpus/agent/test_frozen.jsonl`, fixed since round six, for
facts, songs, weather, compound and small talk; fresh template made tasks for
local and follow ups), run through `bslm.agent.Agent` with the fine tuned
model on llama-server (CPU) and the real tools. "Model alone" judges what the
model delivered; "with the delivery check" judges what the runtime says after
its coded check that a delivered fact appears in the last result and is not
the subject (OWN_MODEL.md stage 6).
Generated by `python -m bslm.agent_bench --limit {args.limit}`; model
`{agent.model.name}`; {len(chosen)} tasks in {(time.time() - t0) / 60:.1f} minutes.

**Overall: {100 * tot_c / max(1, tot_n):.1f}% correct.** Bar per family: 80%.

{chr(10).join(rows)}

Facts by relation: {", ".join(f"{k} {c}/{n}" for k, (c, n) in sorted(per_rel.items())) or "none"}.

## The loop

| metric | value | bar | verdict |
|---|---|---|---|
| recovery rate (correct after a failed first attempt, n={loop['recover_n']}) | {rec:.1f}% | 80% or more | {'pass' if rec >= 80 else '**FAIL**'} |
| false delivery (delivered, wrong, no warning, n={loop['deliver_n']}) | {fal:.1f}% | under 3% | {'pass' if fal < 3 else '**FAIL**'} |
| wasted steps (same action repeated unchanged, over {loop['steps']} steps) | {was:.1f}% | under 10% | {'pass' if was < 10 else '**FAIL**'} |
| honest give up (says so when it cannot be done, n={loop['giveup_n']}) | {giv:.1f}% | 95% or more | {'pass' if giv >= 95 else '**FAIL**'} |

## Misses

{chr(10).join(f"- `{f}` `{u}` -> {a!r} acts {acts}" for f, u, a, acts in misses[:80]) or "- none"}
"""
    OUT.write_text(md, encoding="utf-8", newline="\n")
    print(f"overall {100 * tot_c / max(1, tot_n):.1f}%  families failing: {fails}  recovery {rec:.0f}%  false {fal:.1f}%  wasted {was:.1f}%  giveup {giv:.0f}%")
    print(f"written {OUT}")


if __name__ == "__main__":
    main()
