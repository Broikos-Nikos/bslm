"""The runtime for the trained loop: llama-server produces the model's lines,
this file does the turn taking and runs the actions for real.

    python -m bslm.agent "who directed inception"
    python -m bslm.agent            (interactive)

The model owns Plan, Act, Judge, Ask and Deliver (see pretrain/trajectories.py
for the protocol). Generation stops at the first "Result:" the model would
write, because results come from the environment, never from the model; the
runtime appends the real result and hands the text back. An Ask ends the turn
with a question, a Deliver ends it with the answer.

Memory: the last four turns of the round are kept as one "Earlier:" line
each (the user's words, the last action, what was said; never a result), so
"make it 20 minutes instead" and "and friday?" work. A round ends after ten
quiet minutes or when the user says "new question" or "start over"; that is
a rule here, not a judgement the model makes.

Tools: data/tools.json lists what the wrapper registered, {"name", "desc"}
each; the header carries them and the model calls tool("name", "what").

The delivery check (OWN_MODEL.md stage 6: a coded procedure takes over a
step the model measurably fails): after a search or an open, the delivered
answer must appear in the last result and must not be the subject itself,
otherwise the runtime says it could not confirm it. The benchmark reports
the model alone and the model with the check.

The GGUF is whatever BSLM_GGUF points to, the latest fine tune by default.
"""
import json
import os
import re
import subprocess
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

from pretrain.agent_tools import Env

ROOT = Path(__file__).resolve().parent.parent
SERVER = ROOT / "tools" / "llama" / "llama-server.exe"
MODEL = Path(os.environ.get("BSLM_GGUF", ROOT / "models" / "bslm-72m-v1-q8.gguf"))
PORT = int(os.environ.get("BSLM_LLM_PORT", "8089"))
THREADS = int(os.environ.get("BSLM_LLM_THREADS", "4"))
TOOLS_FILE = ROOT / "data" / "tools.json"
STOP = ["\nResult:", "\nUser:", "<|endoftext|>"]
ROUND_GAP = 600            # seconds of silence that end a round
NEW_ROUND = re.compile(r"^(new (question|topic|round)|start over|forget (that|it|all)|never ?mind)\W*$", re.I)
MEMORY_TURNS = 4
ANSWER_CUT = re.compile(r"^(.+?)(?: is the | wrote | directed | composed | developed | is a | does | metres is | is in | plays )")
QUOTE = re.compile(r'Judge: [^\n]*?"([^"\n]{3,})"')


def norm(s):
    s = unicodedata.normalize("NFD", str(s).lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9 ]+", " ", s).strip()


def load_tools():
    """data/tools.json: [{"name": "qr file receiver", "desc": "receives files sent by QR code",
    "cmd": "C:/path/to/app.exe"}, ...]; cmd is optional and never shown to the model."""
    try:
        return [(t["name"], t.get("desc", ""), t.get("cmd")) for t in json.loads(TOOLS_FILE.read_text(encoding="utf-8"))]
    except (OSError, ValueError, KeyError, TypeError):
        return []


class Agent:
    def __init__(self, env=None, home="Athens", soul="rides a scooter", autostart=True, model=None, port=PORT,
                 tools=None, check=True):
        self.tools = list(tools) if tools is not None else load_tools()
        self.env = env or Env(ROOT / "data" / "state.json", tools=self.tools)
        self.env.tools = list(self.tools)
        self.home, self.soul = home, soul
        self.env.home = home
        self.model = Path(model) if model else MODEL
        self.port = port
        self.check = check
        self.proc = None
        self.transcript = ""
        self.memory = []          # (user text, last action, what was said)
        self.last_time = None
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
             "-c", "4096", "-t", str(THREADS), "-ngl", "0", "--cache-reuse", "256", "--log-disable"],
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
        try:
            r = json.loads(urllib.request.urlopen(req, timeout=180).read())
        except urllib.error.HTTPError as e:
            if e.code == 400:
                return ""          # a prompt the server refuses: the episode ends as a fail
            raise
        return r.get("content", "")

    # ---------------- context ----------------
    def header(self):
        now = self.env.now if self.env.now else datetime.now()
        h = f"Today is {now.strftime('%A %Y-%m-%d, %H:%M')}. Home: {self.home}. Facts: {self.soul}."
        if self.tools:
            h += "\nTools: " + "; ".join(f"{e[0]} ({e[1]})" for e in self.tools)
        for u, act, said in self.memory[-MEMORY_TURNS:]:
            h += f'\nEarlier: "{u}" -> {act or "no action"} -> "{said[:90]}"'
        return h + "\n"

    def new_round(self):
        self.memory = []

    # ---------------- the loop ----------------
    def run(self, text, max_acts=6, resume=False):
        """One user turn. Returns dict(kind, answer, raw_answer, trace, wasted, transcript).
        kind: deliver, ask (the answer is a question back), or fail."""
        text = text.strip()
        now = time.time()
        if not resume:
            if self.last_time and now - self.last_time > ROUND_GAP:
                self.new_round()
            if NEW_ROUND.match(text):
                self.new_round()
                self.last_time = now
                return {"kind": "deliver", "answer": "Fresh start.", "raw_answer": "Fresh start.", "trace": [],
                        "wasted": 0, "transcript": ""}
            self.transcript = self.header()
        self.transcript += "User: " + text + "\n"
        trace, seen_acts, wasted = [], set(), 0
        for _ in range(max_acts + 2):
            out = self.complete(self.transcript)
            kept, action = [], None
            for l in out.split("\n"):
                s = l.strip()
                if not s:
                    continue
                kept.append(s)
                if s.startswith("Deliver:"):
                    self.transcript += "\n".join(kept) + "\n"
                    return self._deliver(text, s[len("Deliver:"):].strip(), trace, wasted, now)
                if s.startswith("Ask:"):
                    self.transcript += "\n".join(kept) + "\n"
                    self.last_time = now
                    q = s[len("Ask:"):].strip()
                    return {"kind": "ask", "answer": q, "raw_answer": q, "trace": trace, "wasted": wasted,
                            "transcript": self.transcript}
                if s.startswith("Act:"):
                    action = s[len("Act:"):].strip()
                    break
            if action is None:
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
                self.last_time = now
                return {"kind": "ask", "answer": q, "raw_answer": q, "trace": trace, "wasted": wasted,
                        "transcript": self.transcript}
        self.last_time = now
        return {"kind": "fail", "answer": "I could not finish that.", "raw_answer": "I could not finish that.",
                "trace": trace, "wasted": wasted, "transcript": self.transcript}

    def _deliver(self, text, answer, trace, wasted, now):
        raw = answer
        unverified = False
        if self.check and any(a.startswith(("search(", "open(")) for a, _ in trace) and not re.search(r"could not|cannot|can't", answer, re.I):
            m = ANSWER_CUT.match(answer)
            cand = m.group(1).strip() if m else ""
            last = trace[-1][1] if trace else ""
            if cand and (norm(cand) not in norm(last) or norm(cand) in norm(text)):
                unverified = True
                answer = "I could not confirm that from the page. What the model read: " + raw
        # the last thing the model quoted from a result: "Wikipedia said this",
        # shown with the answer so a wrong extraction is visible at a glance
        quotes = QUOTE.findall(self.transcript)
        quote = quotes[-1] if quotes and trace else ""
        self.memory.append((text, trace[-1][0] if trace else None, answer))
        self.memory = self.memory[-MEMORY_TURNS:]
        self.last_time = now
        return {"kind": "deliver", "answer": answer, "raw_answer": raw, "unverified": unverified, "trace": trace,
                "quote": quote, "wasted": wasted, "transcript": self.transcript}

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
        print("bslm agent, empty line to quit, 'new question' to forget the round")
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
