"""The runtime for the trained loop: llama-server produces the model's lines,
this file does the turn taking and runs the actions for real.

    python -m bslm.agent "who directed inception"
    python -m bslm.agent            (interactive)

The model owns Plan, Act, Judge, Ask and Deliver (see pretrain/trajectories.py
for the protocol). Generation stops at the first "Result:" the model would
write, because results come from the environment, never from the model; the
runtime appends the real result and hands the text back. An Ask ends the turn
with a question, a Deliver ends it with the answer. The GGUF is whatever
BSLM_GGUF points to, the fine tuned 72m by default.
"""
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

from pretrain.agent_tools import Env

ROOT = Path(__file__).resolve().parent.parent
SERVER = ROOT / "tools" / "llama" / "llama-server.exe"
MODEL = Path(os.environ.get("BSLM_GGUF", ROOT / "models" / "72m-agent-q8.gguf"))
PORT = int(os.environ.get("BSLM_LLM_PORT", "8089"))
THREADS = int(os.environ.get("BSLM_LLM_THREADS", "4"))
STOP = ["\nResult:", "\nUser:", "<|endoftext|>"]


class Agent:
    def __init__(self, env=None, home="Athens", soul="rides a scooter", autostart=True, model=None, port=PORT):
        self.env = env or Env(ROOT / "data" / "state.json")
        self.home, self.soul = home, soul
        self.model = Path(model) if model else MODEL
        self.port = port
        self.proc = None
        self.transcript = ""
        if autostart:
            self.start()

    # ---------------- server ----------------
    def alive(self):
        try:
            return urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2).status == 200
        except Exception:      # noqa: BLE001
            return False

    def start(self, timeout=90):
        if self.alive():
            return
        if not SERVER.exists() or not self.model.exists():
            raise FileNotFoundError(f"missing {SERVER} or {self.model}")
        flags = 0x08000000 if os.name == "nt" else 0
        self.proc = subprocess.Popen(
            [str(SERVER), "-m", str(self.model), "--port", str(self.port), "--host", "127.0.0.1",
             "-c", "2048", "-t", str(THREADS), "-ngl", "0", "--cache-reuse", "256", "--log-disable"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=flags)
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.alive():
                return
            time.sleep(0.5)
        raise RuntimeError("llama-server did not come up")

    def stop(self):
        if self.proc:
            self.proc.terminate()
            self.proc = None

    def complete(self, prompt, max_tokens=160):
        body = json.dumps({"prompt": prompt, "n_predict": max_tokens, "temperature": 0.0, "stop": STOP,
                           "cache_prompt": True, "repeat_penalty": 1.0}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/completion", data=body,
                                     headers={"Content-Type": "application/json"})
        r = json.loads(urllib.request.urlopen(req, timeout=180).read())
        return r.get("content", "")

    # ---------------- the loop ----------------
    def header(self):
        now = self.env.now if self.env.now else datetime.now()
        return f"Today is {now.strftime('%A %Y-%m-%d, %H:%M')}. Home: {self.home}. Facts: {self.soul}.\n"

    def run(self, text, max_acts=6, resume=False):
        """One user turn. Returns dict(kind, answer, trace, steps, transcript).
        kind: deliver, ask (the answer is a question back), or fail."""
        if not resume:
            self.transcript = self.header()
        self.transcript += "User: " + text.strip() + "\n"
        trace, seen_acts, wasted = [], set(), 0
        for _ in range(max_acts + 2):
            out = self.complete(self.transcript)
            lines = [l for l in out.split("\n")]
            kept, action = [], None
            for l in lines:
                s = l.strip()
                if not s:
                    continue
                kept.append(s)
                if s.startswith("Deliver:"):
                    self.transcript += "\n".join(kept) + "\n"
                    return {"kind": "deliver", "answer": s[len("Deliver:"):].strip(), "trace": trace,
                            "wasted": wasted, "transcript": self.transcript}
                if s.startswith("Ask:"):
                    self.transcript += "\n".join(kept) + "\n"
                    return {"kind": "ask", "answer": s[len("Ask:"):].strip(), "trace": trace,
                            "wasted": wasted, "transcript": self.transcript}
                if s.startswith("Act:"):
                    action = s[len("Act:"):].strip()
                    break
            if action is None:
                # the model stopped without acting or answering; nudge once by
                # continuing, then give up
                self.transcript += ("\n".join(kept) + "\n") if kept else "Plan:"
                if not kept:
                    break
                continue
            if action in seen_acts:
                wasted += 1
            seen_acts.add(action)
            result = self.env.act(action)
            trace.append((action, result))
            self.transcript += "\n".join(kept) + "\nResult:\n" + result.strip() + "\n"
            if action.startswith("ask("):
                q = re.sub(r'^ask\((.*)\)$', r"\1", action).strip("\"' ")
                return {"kind": "ask", "answer": q, "trace": trace, "wasted": wasted, "transcript": self.transcript}
        return {"kind": "fail", "answer": "I could not finish that.", "trace": trace, "wasted": wasted,
                "transcript": self.transcript}

    def reply(self, text, max_acts=6):
        """The user's answer to an Ask; continues the same episode."""
        return self.run(text, max_acts=max_acts, resume=True)


def main():
    a = Agent()
    try:
        if len(sys.argv) > 1:
            r = a.run(" ".join(sys.argv[1:]))
            for act, res in r["trace"]:
                print(f"  > {act}\n    {res[:300]}")
            print(r["answer"])
            return
        print("bslm agent, empty line to quit")
        while True:
            try:
                t = input("> ").strip()
            except EOFError:
                break
            if not t:
                break
            r = a.run(t)
            for act, res in r["trace"]:
                print(f"  > {act}\n    {res[:300]}")
            print(("? " if r["kind"] == "ask" else "") + r["answer"])
            while r["kind"] == "ask":
                t = input("  you: ").strip()
                r = a.reply(t)
                for act, res in r["trace"]:
                    print(f"  > {act}\n    {res[:300]}")
                print(("? " if r["kind"] == "ask" else "") + r["answer"])
    finally:
        a.stop()


if __name__ == "__main__":
    main()
