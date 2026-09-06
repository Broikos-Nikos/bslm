"""Byte pair encoding trained from scratch on our own corpus.

Word boundaries are respected: BPE runs inside each pre-token, continuation
pieces carry a '##' prefix. That keeps slot alignment trivial, the first piece
of every word carries the BIO tag and the rest are ignored in the loss.
"""
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

PAD, UNK, CLS = "<pad>", "<unk>", "<cls>"
SPECIALS = [PAD, UNK, CLS]

# the single pre-tokenizer for training data and inference; gen_data imports it.
# apostrophe, typographic apostrophe and acute accent all bind a contraction.
WORD_RE = re.compile("\\w+(?:['\u2019\u00b4]\\w+)?|[^\\w\\s]", re.UNICODE)


def word_tokenize(text):
    """Pre-tokenizer, identical to the one used when the corpus was built."""
    return [(m.group(0), m.start(), m.end()) for m in WORD_RE.finditer(text)]


def normalize(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


def case_id(word):
    if word.isupper() and len(word) > 1:
        return 2
    if word[:1].isupper():
        return 1
    return 0


class BPETokenizer:
    def __init__(self, vocab=None, merges=None):
        self.vocab = vocab or {}                      # token -> id
        self.merges = merges or {}                    # (a, b) -> rank
        self.ids = {i: t for t, i in self.vocab.items()}
        self._cache = {}

    # ---------------- training ----------------
    @classmethod
    def train(cls, texts, vocab_size=4000, min_freq=2, verbose=True):
        freqs = Counter()
        for t in texts:
            for w, _, _ in word_tokenize(normalize(t).lower()):
                freqs[w] += 1

        # every word is a tuple of symbols; '##' marks continuation
        words = {w: tuple([w[0]] + ["##" + c for c in w[1:]]) for w in freqs}
        base = set()
        for sym in words.values():
            base.update(sym)
        vocab = list(SPECIALS) + sorted(base)
        merges = {}
        target = max(0, vocab_size - len(vocab))

        pair_counts = Counter()
        for w, syms in words.items():
            f = freqs[w]
            for i in range(len(syms) - 1):
                pair_counts[(syms[i], syms[i + 1])] += f

        for step in range(target):
            if not pair_counts:
                break
            (a, b), cnt = pair_counts.most_common(1)[0]
            if cnt < min_freq:
                break
            new = a + b[2:] if b.startswith("##") else a + b
            merges[(a, b)] = step
            vocab.append(new)
            # re-apply the merge only where it occurs
            touched = [w for w, s in words.items()
                       if any(s[i] == a and s[i + 1] == b for i in range(len(s) - 1))]
            for w in touched:
                syms, f = words[w], freqs[w]
                for i in range(len(syms) - 1):
                    pair_counts[(syms[i], syms[i + 1])] -= f
                out, i = [], 0
                while i < len(syms):
                    if i < len(syms) - 1 and syms[i] == a and syms[i + 1] == b:
                        out.append(new)
                        i += 2
                    else:
                        out.append(syms[i])
                        i += 1
                syms = tuple(out)
                words[w] = syms
                for i in range(len(syms) - 1):
                    pair_counts[(syms[i], syms[i + 1])] += f
            pair_counts = Counter({p: c for p, c in pair_counts.items() if c > 0})
            if verbose and step % 500 == 0:
                print(f"  merge {step:5d}/{target}  {a}+{b} -> {new}  ({cnt})")

        return cls({t: i for i, t in enumerate(vocab)}, merges)

    # ---------------- encoding ----------------
    def encode_word(self, word):
        word = word.lower()
        if word in self._cache:
            return self._cache[word]
        syms = [word[0]] + ["##" + c for c in word[1:]]
        while len(syms) > 1:
            best, rank = None, None
            for i in range(len(syms) - 1):
                r = self.merges.get((syms[i], syms[i + 1]))
                if r is not None and (rank is None or r < rank):
                    best, rank = i, r
            if best is None:
                break
            a, b = syms[best], syms[best + 1]
            new = a + b[2:] if b.startswith("##") else a + b
            syms = syms[:best] + [new] + syms[best + 2:]
        unk = self.vocab[UNK]
        out = [self.vocab.get(s, unk) for s in syms]
        self._cache[word] = out
        return out

    def encode(self, words, max_len=64):
        """words -> (ids, case_ids, word_index) with a leading <cls>."""
        ids = [self.vocab[CLS]]
        cases = [0]
        widx = [-1]
        for wi, w in enumerate(words):
            pieces = self.encode_word(w)
            c = case_id(w)
            for pi, p in enumerate(pieces):
                if len(ids) >= max_len:
                    break
                ids.append(p)
                cases.append(c)
                widx.append(wi if pi == 0 else -1)
        return ids[:max_len], cases[:max_len], widx[:max_len]

    def encode_text(self, text, max_len=64):
        words = [w for w, _, _ in word_tokenize(normalize(text))]
        return words, self.encode(words, max_len)

    # ---------------- io ----------------
    def save(self, path):
        Path(path).write_text(json.dumps({
            "vocab": self.vocab,
            "merges": [[a, b, r] for (a, b), r in self.merges.items()],
        }, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(d["vocab"], {(a, b): r for a, b, r in d["merges"]})

    def __len__(self):
        return len(self.vocab)
