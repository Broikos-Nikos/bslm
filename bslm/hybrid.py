"""Two layer brain.

reflex     the from scratch 5M router, 4 ms, handles a confident command
reasoning  the 0.6B pretrained model, seconds, handles everything else:
           unsure parses, compound requests, questions it must look up
"""
from .infer import Parser
from .reasoner import Reasoner

REFLEX_THRESHOLD = 0.80


class Hybrid:
    def __init__(self, speak=print):
        self.router = Parser()
        self.reasoner = Reasoner(speak=speak)
        self.bot = self.reasoner.bot                  # one shared state
        self.history = []

    def reply(self, text, force=None):
        """Returns (answer, layer, info)."""
        parsed = self.router.parse(text)
        use_reflex = (force == "reflex") or (
            force is None and parsed["intent"] != "oos"
            and parsed["confidence"] >= REFLEX_THRESHOLD)
        if use_reflex:
            answer = self.bot.run(parsed)
            info = {"intent": parsed["intent"], "confidence": parsed["confidence"],
                    "slots": parsed["slots"]}
            layer = "reflex"
        else:
            answer, info = self.reasoner.run(text, history=self.history[-6:])
            info["router_guess"] = (parsed["raw_intent"], parsed["confidence"])
            layer = "reasoning"
        self.history += [{"role": "user", "content": text},
                         {"role": "assistant", "content": answer}]
        self.history = self.history[-12:]
        return answer, layer, info
