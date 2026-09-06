"""Load the trained BSLM and turn raw text into intent plus slots."""
from pathlib import Path

import torch

from .model import BSLM
from .tokenizer import BPETokenizer, normalize, word_tokenize

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "checkpoints"


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
    return out


class Parser:
    def __init__(self, ckpt=None, tok=None, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(ckpt or (CKPT / "bslm.pt"), map_location=self.device,
                        weights_only=False)
        self.intents, self.tags = ck["intents"], ck["tags"]
        self.max_len = ck["max_len"]
        self.tok = BPETokenizer.load(tok or (CKPT / "tokenizer.json"))
        self.model = BSLM(**ck["cfg"]).to(self.device)
        self.model.load_state_dict(ck["model"])
        self.model.eval()

    @torch.no_grad()
    def parse(self, text, threshold=0.50):
        # 0.50 chosen on 2026-09-06 from a sweep over the command benchmark, the
        # hand written suite and the out of scope suite: against 0.40 it costs
        # 0.7 points on commands, nothing on the suite, and rejects 12.5 points
        # more out of scope text. Above 0.50 nothing improves. Proper
        # calibration and a margin rule are PLAN B4.
        text = normalize(text)
        toks = word_tokenize(text)
        words = [w for w, _, _ in toks]
        if not words:
            return {"text": text, "intent": "oos", "confidence": 1.0, "slots": {}}
        ids, cases, widx = self.tok.encode(words, self.max_len)
        t = lambda x: torch.tensor([x], device=self.device)
        li, ls = self.model(t(ids), t(cases))
        probs = li.softmax(-1)[0]
        conf, idx = probs.max(-1)
        intent = self.intents[idx.item()]
        conf = conf.item()

        pred = ls.argmax(-1)[0].tolist()
        per_word = ["O"] * len(words)
        for pos, w in enumerate(widx):
            if w >= 0 and pos < len(pred):
                per_word[w] = self.tags[pred[pos]]
        slots = {}
        for name, a, b in spans_from_bio(per_word):
            value = text[toks[a][1]:toks[b][2]]
            if name in slots:
                slots[name] = slots[name] if isinstance(slots[name], list) else [slots[name]]
                slots[name].append(value)
            else:
                slots[name] = value
        low = conf < threshold
        return {"text": text, "intent": "oos" if (low or intent == "oos") else intent,
                "raw_intent": intent, "confidence": round(conf, 4),
                "below_threshold": low, "slots": slots,
                "top3": [(self.intents[i], round(probs[i].item(), 3))
                         for i in probs.topk(3).indices.tolist()]}
