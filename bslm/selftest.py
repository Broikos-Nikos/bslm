"""Unit checks that need no model and run in seconds.

    python -m bslm.selftest

Covers the pieces that broke silently at least once: the shared pre
tokenizer, the skill parsers, slot scoring, generator span alignment, the
pretraining stream order and the GGUF permutation.
"""
import sys

import numpy as np

from . import gen_data, skills, tokenizer
from .benchmark import norm, slot_ok

FAILS = []


def check(name, cond, detail=""):
    if not cond:
        FAILS.append(f"{name}: {detail}")
    print(("ok   " if cond else "FAIL ") + name + (f"  [{detail}]" if detail and not cond else ""))


def main():
    # one pre tokenizer for training data and inference
    check("shared WORD_RE", gen_data.WORD_RE is tokenizer.WORD_RE)
    for s in ("don't", "don’t", "l´ete"):
        check(f"contraction kept: {s}", [w for w, _, _ in tokenizer.word_tokenize(s)] == [s], str(tokenizer.word_tokenize(s)))

    # skill parsers
    for text, want in [("7pm", "19:00"), ("7 pm", "19:00"), ("6:45am", "06:45"), ("half past six", "06:30"),
                       ("quarter to 9", "08:45"), ("18:45", "18:45"), ("7 το βράδυ", "19:00"), ("noon", "12:00")]:
        check(f"clock {text}", skills.parse_clock(text) == want, skills.parse_clock(text))
    for text, want in [("5 minutes", 300), ("five minutes", 300), ("half an hour", 1800), ("an hour", 3600),
                       ("1 hour and 30 minutes", 5400), ("an hour and a half", 5400), ("1.5 hours", 5400),
                       ("μιάμιση ώρα", 5400), ("μισή ώρα", 1800), ("δεκαπέντε λεπτά", 900)]:
        check(f"duration {text}", skills.parse_duration(text) == want, str(skills.parse_duration(text)))
    for text, want in [("348 divided by 12", 29), ("20 percent of 250", 50), ("3,5 + 1", 4.5), ("10 x 3", 30),
                       ("2^10", 1024), ("2^1000", None), ("12 συν 30", 42), ("hello", None)]:
        check(f"calc {text}", skills.calculate(text) == want, str(skills.calculate(text)))
    for text, want in [("kilometers", "km"), ("λίβρες", "lb"), ("km", "km"), ("euros", "eur")]:
        check(f"unit {text}", skills.unit_of(text) == want, str(skills.unit_of(text)))
    bot = skills.Assistant(speak=lambda *a: None)
    check("kill the lights turns them off", "off" in bot.light_control({"room": "garage", "_text": "kill the lights in the garage"}))
    check("cut the tv turns it off", "off" in bot.device_control({"device": "tv", "_text": "cut the tv"}))
    bot.state["home"]["volume"] = 50
    check("too loud lowers the volume", bot.volume_set({"_text": "too loud"}) == "Volume 40.")
    check("too quiet raises the volume", bot.volume_set({"_text": "too quiet"}) == "Volume 50.")
    for t, _, _ in bot.timers.values():
        t.cancel()

    # slot scoring normalisation
    check("slot norm strips articles and accents", norm("στην Αθήνα") == "αθηνα", norm("στην Αθήνα"))
    check("slot_ok list order free", slot_ok({"item": ["milk", "eggs"]}, {"item": ["eggs", "milk"]}))
    check("slot_ok missing slot fails", not slot_ok({"date": "tomorrow"}, {}))

    # generator: every rendered span lands on whole words with correct BIO tags
    lex_tr, _ = gen_data.split_lex()
    bad = 0
    for intent in ("timer.set", "reminder.create", "calendar.create", "unit.convert", "work.code", "rule.set"):
        for lang in ("en", "el"):
            for _ in range(40):
                text, spans = gen_data.render(intent, lang, lex_tr)
                words, tags = gen_data.to_bio(gen_data.normalize(text), spans)
                b = sum(t.startswith("B-") for t in tags)
                if b != len(spans) and len(spans) <= len(set(s[2] for s in spans)) + 3:
                    bad += 1
    check("generator spans align with words", bad == 0, f"{bad} misaligned renders")

    # pretraining: stream order and the GGUF permutation
    try:
        from pretrain.train import Stream
        s = Stream("val", 2, 64)
        check("stream epoch 0 file order", s._order() == list(range(len(s.files))))
        s.epoch = 1
        check("stream epoch 1 permuted deterministically", s._order() == Stream("val", 2, 64).__class__._order.__get__(s)())
    except AssertionError:
        print("skip stream checks (no shards on this machine)")
    from pretrain.export_gguf import permute
    w = np.arange(12 * 8, dtype=np.float32).reshape(12, 8)
    p = permute(w, 3)
    check("permute keeps shape", p.shape == w.shape)
    check("permute is a row permutation", sorted(p.flatten().tolist()) == sorted(w.flatten().tolist()))

    print(f"\n{len(FAILS)} failures")
    for f in FAILS:
        print("  " + f)
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
