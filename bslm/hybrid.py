"""Two layer brain.

reflex     the from scratch 5M router, milliseconds, handles a confident
           fixed task command straight through the skills
loop       the from scratch 72M loop model (bslm/agent.py), seconds on the
           CPU, handles everything else: unsure parses, questions it must
           look up, follow ups, tools, weather, songs, the umbrella advice

The loop model carries the memory of the round; the reflex layer's actions
are written into that memory too, so "make it 20 minutes instead" works
after a timer the router set.
"""
import os
import re
from pathlib import Path

from .infer import Parser
from .skills import Assistant

ROOT = Path(__file__).resolve().parent.parent
REFLEX_THRESHOLD = 0.80
# intents the loop model does better than the fixed skill: it reads the real
# forecast, checks the library and the YouTube results, looks facts up
LOOP_INTENTS = {"weather.query", "music.play", "search.web", "knowledge.query", "news.query", "translate.text"}
LEVEL_WORDS = re.compile(r"(half|full|soft|dim|low|high|bright|percent|mode|scene|movie|reading|cozy|relax|party|dinner|focus|sleep)|%", re.I)
DEFAULT_GGUF = ROOT / "models" / "72m-agent7-q8.gguf"


def loop_model():
    p = Path(os.environ.get("BSLM_GGUF", DEFAULT_GGUF))
    return p if p.exists() else None


class Hybrid:
    def __init__(self, speak=print, use_loop=True):
        self.router = Parser()
        self.bot = Assistant(speak=speak)
        self.agent = None
        gguf = loop_model() if use_loop else None
        if gguf:
            from pretrain.agent_tools import Env
            from .agent import Agent, load_tools
            self.agent = Agent(env=Env(ROOT / "data" / "state.json", tools=load_tools()), model=gguf)
            self.agent.env.bot = self.bot            # one shared state between the layers
        self.pending_ask = False

    @property
    def memory(self):
        return self.agent.memory if self.agent else []

    def reply(self, text, force=None):
        """Returns (answer, layer, info). layer: reflex, loop or none."""
        text = text.strip()
        if self.pending_ask and self.agent:
            self.pending_ask = False
            r = self.agent.reply(text)
            self.pending_ask = r["kind"] == "ask"
            return r["answer"], "loop", {"trace": r["trace"], "kind": r["kind"]}
        parsed = self.router.parse(text)
        loop_better = parsed["intent"] in LOOP_INTENTS or (parsed["intent"].startswith("light") and LEVEL_WORDS.search(text))
        use_reflex = (force == "reflex") or (
            force is None and parsed["intent"] != "oos" and parsed["confidence"] >= REFLEX_THRESHOLD
            and not loop_better and not self._looks_like_followup(text))
        if use_reflex or not self.agent:
            answer = self.bot.run(parsed)
            if self.agent:
                self.agent.memory.append((text, f"{parsed['intent']}({', '.join(f'{k}={v}' for k, v in parsed['slots'].items())})", answer))
                self.agent.memory = self.agent.memory[-4:]
            return answer, "reflex", {"intent": parsed["intent"], "confidence": parsed["confidence"], "slots": parsed["slots"]}
        r = self.agent.run(text)
        self.pending_ask = r["kind"] == "ask"
        info = {"trace": r["trace"], "kind": r["kind"], "router_guess": (parsed["raw_intent"], parsed["confidence"]),
                "unverified": r.get("unverified", False)}
        return r["answer"], "loop", info

    def _looks_like_followup(self, text):
        """Short turns that lean on the last one go to the loop model, which
        holds the memory: "and friday?", "make it 20 minutes", "cancel that"."""
        t = text.lower()
        return bool(self.memory) and (len(t.split()) <= 5 or t.startswith(("and ", "also ", "make it", "cancel that", "what about")))
