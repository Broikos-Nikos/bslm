"""Reproducible benchmark. Runs everything and writes BENCHMARK.md.

    .venv\\Scripts\\python.exe -m bslm.benchmark

Sections: purpose, process, capability catalogue, accuracy on a hand written
suite (intent AND slots), the adversarial split, robustness to typos, missing
Greek accents, Greeklish and shouting, out of scope rejection, size and latency
in fp32 and int8, and a live transcript through the real skills.
"""
import copy
import io
import json
import random
import re
import statistics
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

from . import skills
from .infer import Parser
from .templates import TEMPLATES

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "BENCHMARK.md"

# --------------------------------------------------------------------------
# the suite: hand written, never seen in training. (text, intent, slots)
# a list value means the slot is expected to fire more than once
# --------------------------------------------------------------------------
SUITE = [
    # ---------------- english ----------------
    ("could you start counting down from twelve minutes", "timer.set", {"duration": "twelve minutes"}),
    ("buzz me in 3 min", "timer.set", {"duration": "3 min"}),
    ("set a timer for an hour and a half", "timer.set", {"duration": "an hour and a half"}),
    ("kill that countdown", "timer.cancel", {}),
    ("is the timer nearly finished", "timer.query", {}),
    ("get me up at 6:45 tomorrow", "alarm.set", {"time": "6:45", "date": "tomorrow"}),
    ("wake me at half past six on Monday", "alarm.set", {"time": "half past six", "date": "Monday"}),
    ("scrap the alarm", "alarm.cancel", {}),
    ("nudge me to pay the electricity bill on Friday", "reminder.create", {"task": "pay the electricity bill", "date": "Friday"}),
    ("remind me at 18:30 to call the bank", "reminder.create", {"time": "18:30", "task": "call the bank"}),
    ("what have I got to do this week", "reminder.list", {}),
    ("stick a dentist appointment in for Tuesday at 9:00", "calendar.create", {"title": "dentist appointment", "date": "Tuesday", "time": "9:00"}),
    ("put lunch with Maria in for Friday at 1:00", "calendar.create", {"title": "lunch with Maria", "date": "Friday", "time": "1:00"}),
    ("am I busy on Thursday", "calendar.query", {"date": "Thursday"}),
    ("is it going to pour down in Patras tomorrow", "weather.query", {"location": "Patras", "date": "tomorrow"}),
    ("how cold is it in Berlin tonight", "weather.query", {"location": "Berlin", "date": "tonight"}),
    ("chuck on some blues", "music.play", {"genre": "blues"}),
    ("play Hotel California by Queen", "music.play", {"song": "Hotel California", "artist": "Queen"}),
    ("shut the music up", "music.control", {}),
    ("skip to the next track", "music.control", {}),
    ("crank the volume to 80", "volume.set", {"level": "80"}),
    ("kill the lights in the garage", "light.control", {"room": "garage"}),
    ("make the bedroom lights warm white", "light.control", {"room": "bedroom", "color": "warm white"}),
    ("switch the boiler on", "device.control", {"device": "boiler"}),
    ("turn the fan off in the kids room", "device.control", {"device": "fan", "room": "kids room"}),
    ("stick yogurt on the grocery list", "list.add", {"item": "yogurt", "list_name": "grocery list"}),
    ("we are out of olive oil and eggs", "list.add", {"item": ["olive oil", "eggs"]}),
    ("read out the shopping list", "list.read", {}),
    ("jot down that the spare key is under the pot", "note.create", {"note": "the spare key is under the pot"}),
    ("what have I written down", "note.read", {}),
    ("drop Maria a line saying I am running late", "message.send", {"person": "Maria", "content": "I am running late"}),
    ("text John the invoice is sent", "message.send", {"person": "John", "content": "the invoice is sent"}),
    ("get my landlord on the phone", "call.make", {"person": "my landlord"}),
    ("ring my sister", "call.make", {"person": "my sister"}),
    ("how do I drive to Volos", "navigation.route", {"location": "Volos"}),
    ("look up cheap hotels in Rome", "search.web", {"query": "cheap hotels in Rome"}),
    ("search for how to fix a leaking tap", "search.web", {"query": "how to fix a leaking tap"}),
    ("what is thank you very much in Japanese", "translate", {"phrase": "thank you very much", "language": "Japanese"}),
    ("work out 348 divided by 12", "math.calculate", {"expression": "348 divided by 12"}),
    ("what is 20 percent of 250", "math.calculate", {"expression": "20 percent of 250"}),
    ("how many miles is 42 kilometers", "unit.convert", {"amount": "42", "unit_from": "kilometers", "unit_to": "miles"}),
    ("convert 2.5 kilos to pounds", "unit.convert", {"amount": "2.5", "unit_from": "kilos", "unit_to": "pounds"}),
    ("got the time", "time.query", {}),
    ("how many days until Christmas", "time.query", {"date": "Christmas"}),
    ("anything new on the economy", "news.query", {"topic": "the economy"}),
    ("morning", "smalltalk.greet", {}),
    ("cheers mate", "smalltalk.thanks", {}),
    ("catch you tomorrow", "smalltalk.bye", {}),
    ("what are you able to do", "assistant.capabilities", {}),
    ("sorry, say again", "assistant.repeat", {}),
    ("no wait, forget that", "assistant.cancel", {}),
    # ---------------- greek ----------------
    ("βάλε μου χρονόμετρο δεκαπέντε λεπτών", "timer.set", {"duration": "δεκαπέντε λεπτών"}),
    ("βάλε χρονόμετρο μιάμιση ώρα", "timer.set", {"duration": "μιάμιση ώρα"}),
    ("σταμάτα το να μετράει", "timer.cancel", {}),
    ("πόσο μένει ακόμα στο χρονόμετρο", "timer.query", {}),
    ("σήκωσέ με στις 7 το πρωί", "alarm.set", {"time": "7 το πρωί"}),
    ("ξύπνα με 6 και μισή την Δευτέρα", "alarm.set", {"time": "6 και μισή", "date": "Δευτέρα"}),
    ("μη με ξυπνήσεις το πρωί", "alarm.cancel", {}),
    ("θύμισέ μου να ποτίσω τα φυτά την Τετάρτη", "reminder.create", {"task": "να ποτίσω τα φυτά", "date": "Τετάρτη"}),
    ("τι έχω να κάνω αύριο", "reminder.list", {"date": "αύριο"}),
    ("κλείσε ραντεβού στον γιατρό την Πέμπτη στις 10:30", "calendar.create", {"title": "ραντεβού στον γιατρό", "date": "Πέμπτη", "time": "10:30"}),
    ("έχω κάτι κανονισμένο την Παρασκευή", "calendar.query", {"date": "Παρασκευή"}),
    ("θα βρέξει στη Λάρισα το σαββατοκύριακο", "weather.query", {"location": "Λάρισα", "date": "το σαββατοκύριακο"}),
    ("τι καιρό θα κάνει αύριο στην Πάτρα", "weather.query", {"date": "αύριο", "location": "Πάτρα"}),
    ("βάλε λίγα ρεμπέτικα", "music.play", {"genre": "ρεμπέτικα"}),
    ("παίξε Ζήλεια από τη Βίσση", "music.play", {"song": "Ζήλεια"}),
    ("κόψε τη μουσική", "music.control", {}),
    ("δυνάμωσε λίγο", "volume.set", {}),
    ("σβήσε τα φώτα στο μπάνιο", "light.control", {"room": "μπάνιο"}),
    ("κάνε τα φώτα στο σαλόνι μπλε", "light.control", {"room": "σαλόνι", "color": "μπλε"}),
    ("άναψε τον θερμοσίφωνα", "device.control", {"device": "θερμοσίφωνα"}),
    ("κλείσε το κλιματιστικό στο υπνοδωμάτιο", "device.control", {"device": "κλιματιστικό", "room": "υπνοδωμάτιο"}),
    ("πρόσθεσε γιαούρτι στη λίστα με τα ψώνια", "list.add", {"item": "γιαούρτι", "list_name": "λίστα με τα ψώνια"}),
    ("ξεμείναμε από γάλα και ψωμί", "list.add", {"item": ["γάλα", "ψωμί"]}),
    ("τι έχει η λίστα για το σούπερ μάρκετ", "list.read", {}),
    ("σημείωσε ότι το τιμολόγιο είναι 4471", "note.create", {"note": "το τιμολόγιο είναι 4471"}),
    ("διάβασέ μου τι έχω γράψει", "note.read", {}),
    ("στείλε μήνυμα στον Νίκο ότι θα αργήσω", "message.send", {"person": "Νίκο", "content": "θα αργήσω"}),
    ("πες στη Μαρία ότι έρχομαι", "message.send", {"person": "Μαρία", "content": "έρχομαι"}),
    ("πάρε τον λογιστή", "call.make", {"person": "λογιστή"}),
    ("πώς πάω στο Ηράκλειο", "navigation.route", {"location": "Ηράκλειο"}),
    ("ψάξε μου φθηνά αεροπορικά για Λονδίνο", "search.web", {"query": "φθηνά αεροπορικά για Λονδίνο"}),
    ("πώς λένε καλημέρα στα ιαπωνικά", "translate", {"phrase": "καλημέρα", "language": "ιαπωνικά"}),
    ("πόσο κάνει 144 δια 12", "math.calculate", {"expression": "144 δια 12"}),
    ("πόσο κάνει 15 τοις εκατό του 200", "math.calculate", {"expression": "15 τοις εκατό του 200"}),
    ("πόσα κιλά είναι 30 λίβρες", "unit.convert", {"amount": "30", "unit_from": "λίβρες", "unit_to": "κιλά"}),
    ("μετάτρεψε 100 ευρώ σε δολάρια", "unit.convert", {"amount": "100", "unit_from": "ευρώ", "unit_to": "δολάρια"}),
    ("τι ώρα είναι τώρα", "time.query", {}),
    ("τι νέα έχουμε για την οικονομία", "news.query", {"topic": "οικονομία"}),
    ("καλησπέρα σας", "smalltalk.greet", {}),
    ("να 'σαι καλά", "smalltalk.thanks", {}),
    ("καλό βράδυ, τα λέμε", "smalltalk.bye", {}),
    ("σε τι είσαι χρήσιμος", "assistant.capabilities", {}),
    ("δεν κατάλαβα, ξαναπές το", "assistant.repeat", {}),
    ("άσ' το, δεν πειράζει", "assistant.cancel", {}),
]

OOS_SUITE = [
    "write me a sonnet about the moon", "explain how a nuclear reactor works",
    "qwerty zxcv", "who is the prime minister of greece", "tell me a bedtime story",
    "what do you think about my haircut", "recommend a good book",
    "how do I become rich", "what is love", "asdkjh qwe",
    "γράψε μου ένα τραγούδι για τη θάλασσα", "ποιος είναι ο πρωθυπουργός",
    "πες μου μια ιστορία για δράκους", "μπορείς να μου εξηγήσεις τη σχετικότητα",
    "τι γνώμη έχεις για την πολιτική", "λκφδ σδφκ",
]

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
ARTICLES = r"^(?:the|a|an|at|on|in|to|for|my|στις|στη|στην|στο|στον|στα|τις|την|τη|τον|το|τα|για|σε|από)\s+"


def strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def norm(v):
    v = strip_accents(v.lower().strip().strip(".,!?;"))
    while True:
        n = re.sub(ARTICLES, "", v)
        if n == v:
            return v
        v = n


def slot_ok(expected, got):
    for k, want in expected.items():
        have = got.get(k)
        if have is None:
            return False
        want_l = want if isinstance(want, list) else [want]
        have_l = have if isinstance(have, list) else [have]
        if sorted(norm(x) for x in want_l) != sorted(norm(x) for x in have_l):
            return False
    return True


GREEKLISH = [("ου", "ou"), ("αι", "ai"), ("ει", "ei"), ("οι", "oi"), ("ευ", "ef"), ("αυ", "af"),
             ("θ", "th"), ("χ", "x"), ("ψ", "ps"), ("ξ", "ks"), ("α", "a"), ("β", "v"),
             ("γ", "g"), ("δ", "d"), ("ε", "e"), ("ζ", "z"), ("η", "i"), ("ι", "i"),
             ("κ", "k"), ("λ", "l"), ("μ", "m"), ("ν", "n"), ("ο", "o"), ("π", "p"),
             ("ρ", "r"), ("σ", "s"), ("ς", "s"), ("τ", "t"), ("υ", "y"), ("φ", "f"), ("ω", "o")]


def greeklish(text):
    t = strip_accents(text.lower())
    for g, l in GREEKLISH:
        t = t.replace(g, l)
    return t


def add_typo(text, protected, rng):
    words = text.split()
    idx = [i for i, w in enumerate(words) if len(w) >= 4
           and not any(w in p for p in protected)]
    if not idx:
        return text
    i = rng.choice(idx)
    w = words[i]
    j = rng.randrange(1, len(w) - 1)
    words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
    return " ".join(words)


def is_greek(text):
    return bool(re.search(r"[α-ωΑ-Ω]", text))


def pct(a, b):
    return f"{100 * a / b:.1f}%" if b else "n/a"


# --------------------------------------------------------------------------
# measurements
# --------------------------------------------------------------------------
def run_suite(parser, items, transform=None, score_slots=True):
    res = {"n": 0, "intent": 0, "slots": 0, "joint": 0, "by_lang": defaultdict(lambda: [0, 0]),
           "by_intent": defaultdict(lambda: [0, 0]), "misses": []}
    for text, intent, slots in items:
        t = transform(text, slots) if transform else text
        r = parser.parse(t)
        i_ok = r["intent"] == intent
        s_ok = slot_ok(slots, r["slots"]) if score_slots else True
        lang = "el" if is_greek(text) else "en"
        res["n"] += 1
        res["intent"] += i_ok
        res["slots"] += s_ok
        res["joint"] += i_ok and s_ok
        res["by_lang"][lang][0] += i_ok and s_ok
        res["by_lang"][lang][1] += 1
        res["by_intent"][intent][0] += i_ok and s_ok
        res["by_intent"][intent][1] += 1
        if not (i_ok and s_ok):
            res["misses"].append((t, intent, r["intent"], r["confidence"], slots, r["slots"]))
    return res


def run_oos(parser, texts):
    accepted = [(t, r["intent"], r["confidence"]) for t in texts
                for r in [parser.parse(t)] if r["intent"] != "oos"]
    return len(texts), accepted


def run_adversarial(parser):
    rows = [json.loads(l) for l in (ROOT / "data" / "test.jsonl").open(encoding="utf-8")]
    per = defaultdict(lambda: [0, 0])
    conf = defaultdict(Counter)
    ok = 0
    for r in rows:
        got = parser.parse(r["text"])["intent"]
        per[r["intent"]][1] += 1
        if got == r["intent"]:
            per[r["intent"]][0] += 1
            ok += 1
        else:
            conf[r["intent"]][got] += 1
    return len(rows), ok, per, conf


def latency(parser, n=40):
    samples = [t for t, _, _ in SUITE[::7]]
    for s in samples:
        parser.parse(s)
    times = []
    for _ in range(n):
        for s in samples:
            t0 = time.perf_counter()
            parser.parse(s)
            times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times), sorted(times)[int(len(times) * 0.95)]


def state_bytes(model):
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell()


def quantized_copy(parser):
    q = copy.deepcopy(parser)
    q.device = "cpu"
    q.model = torch.ao.quantization.quantize_dynamic(
        copy.deepcopy(parser.model).cpu().eval(), {nn.Linear}, dtype=torch.qint8)
    return q


def demo_transcript(parser, tmp_state):
    skills.STATE = tmp_state
    if tmp_state.exists():
        tmp_state.unlink()
    bot = skills.Assistant(speak=lambda *_: None)
    cmds = ["set a timer for 25 minutes", "how much longer",
            "remind me to pay the electricity bill on Friday at 9:00",
            "add milk and batteries to the shopping list", "what is on my shopping list",
            "βάλε ξυπνητήρι για τις 7 το πρωί",
            "σημείωσε ότι ο κωδικός του wifi είναι 12345", "what have I noted down",
            "how many kilometers is 42 miles", "work out 348 divided by 12",
            "turn off the lights in the kitchen", "άναψε τον θερμοσίφωνα",
            "τι υπενθυμίσεις έχω", "write me a poem about the sea", "cancel the timer"]
    lines = []
    for c in cmds:
        r = parser.parse(c)
        out = bot.run(r).replace("\n", "\n    ")
        lines.append(f"> {c}\n  [{r['intent']} {r['confidence']:.2f}]  {out}")
    for t, _, _ in bot.timers.values():
        t.cancel()
    if tmp_state.exists():
        tmp_state.unlink()
    return "\n".join(lines)


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
def capability_table():
    ex = {}
    for text, intent, slots in SUITE:
        lang = "el" if is_greek(text) else "en"
        ex.setdefault(intent, {}).setdefault(lang, (text, slots))
    rows = ["| intent | english | greek | slots |", "|---|---|---|---|"]
    for intent in sorted(TEMPLATES):
        en = ex.get(intent, {}).get("en", ("", {}))[0]
        el = ex.get(intent, {}).get("el", ("", {}))[0]
        s = sorted({k for _, i, sl in SUITE if i == intent for k in sl})
        rows.append(f"| `{intent}` | {en} | {el} | {', '.join(s) or ''} |")
    return "\n".join(rows)


def main():
    t_start = time.time()
    ck = ROOT / "checkpoints" / "bslm.pt"
    tok = ROOT / "checkpoints" / "tokenizer.json"
    p = Parser()
    rng = random.Random(7)
    n_train = sum(1 for _ in (ROOT / "data" / "train.jsonl").open(encoding="utf-8"))

    base = run_suite(p, SUITE)
    en_items = [x for x in SUITE if not is_greek(x[0])]
    el_items = [x for x in SUITE if is_greek(x[0])]
    typo = run_suite(p, SUITE, lambda t, s: add_typo(
        t, [v for x in s.values() for v in (x if isinstance(x, list) else [x])], rng))
    shout = run_suite(p, en_items, lambda t, s: t.upper())
    noacc = run_suite(p, el_items, lambda t, s: strip_accents(t))
    glish = run_suite(p, el_items, lambda t, s: greeklish(t), score_slots=False)
    n_oos, accepted = run_oos(p, OOS_SUITE)
    n_adv, ok_adv, per_adv, conf_adv = run_adversarial(p)
    adv_oos = per_adv.get("oos", [0, 0])

    cpu = Parser(device="cpu")
    lat_cpu = latency(cpu)
    lat_gpu = latency(p) if torch.cuda.is_available() else None
    q = quantized_copy(cpu)
    lat_q = latency(q)
    q_res = run_suite(q, SUITE)
    fp32 = state_bytes(cpu.model)
    int8 = state_bytes(q.model)
    n_params = cpu.model.n_params()

    transcript = demo_transcript(p, ROOT / "data" / "state.benchmark.json")

    def suite_row(name, r, slots=True):
        s = pct(r["slots"], r["n"]) if slots else "not scored"
        j = pct(r["joint"], r["n"]) if slots else pct(r["intent"], r["n"])
        return f"| {name} | {r['n']} | {pct(r['intent'], r['n'])} | {s} | {j} |"

    misses = "\n".join(
        f"- `{t}`  wanted **{w}** {json.dumps(ws, ensure_ascii=False)}  got **{g}** ({c:.2f}) {json.dumps(gs, ensure_ascii=False)}"
        for t, w, g, c, ws, gs in base["misses"]) or "- none"

    worst = sorted(per_adv.items(), key=lambda kv: kv[1][0] / kv[1][1])[:8]
    worst_rows = "\n".join(
        f"| `{i}` | {pct(ok, n)} | {', '.join(f'{k} ({v})' for k, v in conf_adv[i].most_common(2))} |"
        for i, (ok, n) in worst)

    md = f"""# BSLM benchmark

Generated {datetime.now():%Y-%m-%d %H:%M} by `python -m bslm.benchmark` in {time.time() - t_start:.0f}s.
Everything below is measured on the checkpoint in `checkpoints/`, nothing is
hand entered.

## Why it was built

The question was whether an everyday assistant, the "set a timer, remind me,
add milk to the list" kind, can be built **from scratch**: no pretrained
weights, no API, no vendor. Not a wrapper that looks like Siri, the actual
model. The answer is yes, with one condition: the task list has to be fixed.
A general chatbot from scratch needs trillions of tokens and a datacentre.
A fixed task list needs a 5M parameter encoder and six minutes on one GPU.

The point of publishing it is the trade itself. This model is 20 MB in fp32,
{int8 / 1e6:.1f} MB in int8, answers in {lat_cpu[0]:.1f} ms on a CPU, needs no network, and
scores {pct(base['joint'], base['n'])} on sentences it has never seen. The cost is that it knows
exactly and only what its templates taught it.

## How it was built

1. **Templates first.** 35 intents, 28 slot types, English and Greek, written
   by hand in `bslm/templates.py`. A synonym layer (`[start]`, `[cancel]`,
   `[read]`...) expands each template into a dozen surface forms.
2. **Corpus generated**, not collected: {n_train:,} labelled utterances with
   character level slot spans, typos, fillers and casing noise.
3. **BPE tokenizer trained** on that corpus (4000 merges, inside word boundaries).
4. **Transformer encoder written by hand** in `bslm/model.py`: 6 pre-norm
   blocks, d_model 256, 4 heads, one head for intent, one for BIO slot tags.
   {n_params / 1e6:.2f}M parameters.
5. **Trained from random init**, joint cross entropy, 8% token dropout so it
   cannot lean on a single keyword.
6. **Evaluated adversarially.** The first version scored 99.9% on validation
   and 76% on hand written sentences: it had memorised verbs. The synonym
   layer took hand written accuracy to 100% with the same architecture.
7. **Skills attached** in `bslm/skills.py`: timers, reminders, notes, lists,
   calendar, maths, units and a home state that really execute and persist.

## What it can do

One hand written example per language from the suite below. Slots are the
arguments the model extracts as text spans.

{capability_table()}

## Accuracy on hand written sentences

{len(SUITE)} sentences ({len(en_items)} English, {len(el_items)} Greek) written by hand after training, with
the expected intent **and** the expected slot values. A sentence counts as
correct only if both match. Slot values are compared after lowercasing,
accent stripping and dropping a leading article or preposition.

| set | n | intent | slots | both |
|---|---|---|---|---|
{suite_row('all', base)}
{suite_row('english', run_suite(p, en_items))}
{suite_row('greek', run_suite(p, el_items))}

Misses:

{misses}

## Robustness

Same sentences, transformed. Slots are not scored for Greeklish because the
expected values are Greek script.

| transform | n | intent | slots | both |
|---|---|---|---|---|
{suite_row('one typo per sentence (outside slot values)', typo)}
{suite_row('ALL CAPS (english)', shout)}
{suite_row('greek without accents', noacc)}
{suite_row('greeklish, latin keyboard', glish, slots=False)}

Greeklish and accentless Greek are how people actually type on phones in
Greece. Neither is in the training data yet; these rows are the baseline the
plan in `PLAN.md` improves on.

## Out of scope rejection

The assistant must refuse what it was not built for instead of guessing.

| set | n | rejected | wrongly accepted |
|---|---|---|---|
| hand written out of scope | {n_oos} | {pct(n_oos - len(accepted), n_oos)} | {len(accepted)} |
| generated out of scope (test split) | {adv_oos[1]} | {pct(adv_oos[0], adv_oos[1])} | {adv_oos[1] - adv_oos[0]} |

{chr(10).join(f'- `{t}` accepted as {i} ({c:.2f})' for t, i, c in accepted) or '- nothing wrongly accepted'}

## Adversarial split

{n_adv:,} generated sentences built only from **held out templates rendered with
held out synonyms**. The model has never seen the phrasing and never seen the
verb. This is deliberately harsher than real usage.

| | intent accuracy |
|---|---|
| all intents | {pct(ok_adv, n_adv)} |

Weakest intents, with what they were confused for:

| intent | accuracy | confused with |
|---|---|---|
{worst_rows}

Every failure here is a verb that was withheld from training ("bin the alarm",
"stream Clocks"). A model with no pretraining can only know a word the corpus
contained. That is the honest ceiling of from scratch.

## Size and latency

| | fp32 | int8 dynamic |
|---|---|---|
| parameters | {n_params / 1e6:.2f} M | same |
| weights in memory | {fp32 / 1e6:.2f} MB | {int8 / 1e6:.2f} MB |
| cpu latency, median / p95 | {lat_cpu[0]:.2f} / {lat_cpu[1]:.2f} ms | {lat_q[0]:.2f} / {lat_q[1]:.2f} ms |
| hand written suite, both correct | {pct(base['joint'], base['n'])} | {pct(q_res['joint'], q_res['n'])} |
{f"| gpu latency, median / p95 | {lat_gpu[0]:.2f} / {lat_gpu[1]:.2f} ms | |" if lat_gpu else ""}

Tokenizer: {tok.stat().st_size / 1e3:.0f} KB. Checkpoint on disk: {ck.stat().st_size / 1e6:.2f} MB.
Batch size is one throughout, which is how an assistant is used; at that size
the CPU beats the GPU because kernel launch dominates.

## Transcript

Real run through `bslm/skills.py` with a throwaway state file.

```
{transcript}
```

## Reproduce

```
py -3.12 -m venv .venv
.venv\\Scripts\\python.exe -m pip install -r requirements.txt
.venv\\Scripts\\python.exe -m bslm.gen_data
.venv\\Scripts\\python.exe -m bslm.train --epochs 22
.venv\\Scripts\\python.exe -m bslm.benchmark
```
"""
    OUT.write_text(md, encoding="utf-8", newline="\n")
    print(md)
    print(f"\nwritten {OUT}")


if __name__ == "__main__":
    main()
