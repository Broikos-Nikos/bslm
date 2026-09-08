"""Run the shipped v1 model through a scripted session and write a real
transcript to DEMO.md, so a reader can see the from-scratch loop
model actually search, read, retry, use tools and hold a short memory.
Every Result block is produced live (Wikipedia, Open-Meteo, YouTube, the
local skills); nothing here is mocked."""
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("BSLM_GGUF", str(ROOT / "models" / "bslm-72m-v1-q8.gguf"))

from bslm.agent import Agent
from pretrain.agent_tools import Env

# (label, prompt, continues_previous_round)
SCRIPT = [
    ("A fact it looks up", "who directed the film Inception", False),
    ("The weather, read from a live forecast", "whats the weather tomorrow in Athens", False),
    ("A timer through the clock app", "set a timer for 10 minutes", False),
    ("Changing it, memory of the round plus a retry", "make it 20 minutes instead", True),
    ("A song, library first then YouTube", "play bohemian rhapsody by queen", False),
    ("Smart home, a light level", "set the living room lights to half", False),
    ("A registered tool", "open the qr file receiver", False),
    ("Something it should refuse", "write me a poem about the sea", False),
    ("A fact it cannot find, honest give up", "who is the head coach of the Fictropolis Rovers", False),
    ("A limitation, shown honestly (a two hop question it was not trained for)", "who directed inception", False),
    ("  the second hop reuses the first answer's shape instead of composing", "and who composed the music for it", True),
]


def main():
    env = Env(ROOT / "data" / "demo_state.json", tools=[("qr file receiver", "receives files sent by QR code", None)])
    a = Agent(env=env, model=str(Path(os.environ["BSLM_GGUF"])), tools=[("qr file receiver", "receives files sent by QR code", None)], port=8092)
    out = ["# BSLM v1, a live session",
           "",
           "The from-scratch 72M loop model (`bslm-72m-v1-q8.gguf`, 78 MB, Q8_0) driving",
           "the real tools through `bslm/agent.py`. Every `>` line is an action the model",
           "chose and the runtime ran for real (Wikipedia, Open-Meteo, YouTube, the local",
           "skills); the reply is what it delivered. Captured by `bslm/demo.py`,",
           "nothing mocked. The last section is kept in on purpose to show a real",
           "limitation; the honest per-family scores are in `AGENT_BENCHMARK.md` (91% overall,",
           "open fact lookup about 70%, every other family passing).", ""]
    try:
        for label, prompt, cont in SCRIPT:
            r = a.reply(prompt) if cont else a.run(prompt)
            out.append(f"### {label}")
            out.append(f"**You:** {prompt}  ")
            for act, res in r["trace"]:
                res1 = res.replace("\n", " ")[:150]
                out.append(f"> `{act}` -> {res1}")
            tag = "?" if r["kind"] == "ask" else ""
            out.append(f"**Bee:** {tag}{r['answer']}")
            out.append(f"_{len(r['trace'])} action(s), {r.get('seconds', 0):.1f}s_")
            out.append("")
            print(f"{label}: {r['kind']} ({len(r['trace'])} acts)", flush=True)
    finally:
        a.stop()
    (ROOT / "DEMO.md").write_text("\n".join(out), encoding="utf-8", newline="\n")
    print("wrote DEMO.md")


if __name__ == "__main__":
    main()
