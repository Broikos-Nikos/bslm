"""What the assistant actually does once the model has understood.

Everything here is local and offline. Timers, reminders, notes, lists,
calendar, maths, unit conversion and the home state really work and persist.
Skills that need a third party service report the exact call they would make,
which is the honest boundary of a from scratch NLU model.
"""
import json
import os
import re
import threading
import time
import unicodedata
import webbrowser
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import quote_plus


def _open(url):
    """Open a page in the default browser. BSLM_NO_BROWSER=1 disables it (tests)."""
    if os.environ.get("BSLM_NO_BROWSER"):
        return False
    try:
        return webbrowser.open(url)
    except Exception:
        return False


LANG_CODES = {
    "greek": "el", "english": "en", "german": "de", "french": "fr", "spanish": "es",
    "italian": "it", "japanese": "ja", "turkish": "tr", "russian": "ru",
    "portuguese": "pt", "ελληνικα": "el", "αγγλικα": "en", "γερμανικα": "de",
    "γαλλικα": "fr", "ισπανικα": "es", "ιταλικα": "it", "ιαπωνικα": "ja",
    "τουρκικα": "tr", "ρωσικα": "ru", "πορτογαλικα": "pt",
}

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "data" / "state.json"

NUM_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40, "forty five": 45,
    "fifty": 50, "sixty": 60, "ninety": 90, "a": 1, "an": 1, "half": 0.5,
    "ενα": 1, "δυο": 2, "τρια": 3, "τεσσερα": 4, "πεντε": 5, "εξι": 6,
    "εφτα": 7, "επτα": 7, "οκτω": 8, "οχτω": 8, "εννια": 9, "δεκα": 10,
    "δεκαπεντε": 15, "εικοσι": 20, "τριαντα": 30, "σαραντα": 40,
    "σαραντα πεντε": 45, "πενηντα": 50, "εξηντα": 60, "ενενηντα": 90,
    "μιση": 0.5, "μισο": 0.5, "μια": 1, "ενας": 1,
}

SEC_UNITS = [
    (r"\b(sec|second|seconds|δευτ|δευτερολεπτ)\w*", 1),
    (r"\b(min|minute|minutes|λεπτ)\w*", 60),
    (r"\b(hour|hours|hr|ωρ)\w*", 3600),
    (r"\b(day|days|μερ|ημερ)\w*", 86400),
]

CONV = {
    ("km", "mi"): 0.621371, ("mi", "km"): 1.609344,
    ("kg", "lb"): 2.204623, ("lb", "kg"): 0.453592,
    ("m", "ft"): 3.28084, ("ft", "m"): 0.3048,
    ("cm", "in"): 0.393701, ("in", "cm"): 2.54,
    ("l", "gal"): 0.264172, ("gal", "l"): 3.785412,
    ("eur", "usd"): 1.08, ("usd", "eur"): 0.926,
    ("g", "oz"): 0.035274, ("oz", "g"): 28.3495,
}
UNIT_ALIAS = {
    "km": "km", "kilometer": "km", "kilometers": "km", "kilometres": "km",
    "χιλιομετρα": "km", "χλμ": "km",
    "mi": "mi", "mile": "mi", "miles": "mi", "μιλια": "mi",
    "kg": "kg", "kilo": "kg", "kilos": "kg", "kilograms": "kg", "κιλα": "kg",
    "lb": "lb", "lbs": "lb", "pound": "lb", "pounds": "lb", "λιβρες": "lb",
    "m": "m", "meter": "m", "meters": "m", "metres": "m", "μετρα": "m",
    "ft": "ft", "foot": "ft", "feet": "ft", "ποδια": "ft",
    "cm": "cm", "centimeter": "cm", "centimeters": "cm", "εκατοστα": "cm",
    "in": "in", "inch": "in", "inches": "in", "ιντσες": "in",
    "l": "l", "liter": "l", "liters": "l", "litres": "l", "λιτρα": "l",
    "gal": "gal", "gallon": "gal", "gallons": "gal", "γαλονια": "gal",
    "eur": "eur", "euro": "eur", "euros": "eur", "ευρω": "eur",
    "usd": "usd", "dollar": "usd", "dollars": "usd", "δολαρια": "usd",
    "g": "g", "gram": "g", "grams": "g", "γραμμαρια": "g",
    "oz": "oz", "ounce": "oz", "ounces": "oz", "ουγγιες": "oz",
    "celsius": "c", "κελσιου": "c", "c": "c",
    "fahrenheit": "f", "φαρεναιτ": "f", "f": "f",
}


def strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFD", s.lower())
                   if unicodedata.category(c) != "Mn")


def load_state():
    if STATE.exists():
        return json.loads(STATE.read_text(encoding="utf-8"))
    return {"reminders": [], "notes": [], "lists": {}, "calendar": [],
            "alarms": [], "home": {"lights": {}, "devices": {}, "volume": 40},
            "music": {"playing": None, "paused": False}}


def save_state(s):
    STATE.parent.mkdir(exist_ok=True)
    STATE.write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------- parsing
def _unit_mult(word):
    for pat, mult in SEC_UNITS:
        if re.match(pat, word):
            return mult
    return None


SPECIAL_DUR = [
    (r"an hour and a half|\bμιαμιση ωρα", 5400),
    (r"half an hour|\bμιση ωρα", 1800),
    (r"half a minute|μισο λεπτο", 30),
    (r"a minute and a half|εναμιση λεπτο", 90),
]

_NUM_ALT = "|".join(re.escape(w) for w in sorted(NUM_WORDS, key=len, reverse=True))
_PAIR_RE = re.compile(r"(\d+(?:[.,]\d+)?|\b(?:" + _NUM_ALT + r")\b)\s+([^\W\d_]+)")


def parse_duration(text):
    """Sum every (number, unit) pair: '1 hour and 30 minutes' -> 5400."""
    if not text:
        return None
    t = strip_accents(text)
    for pat, secs in SPECIAL_DUR:
        if re.search(pat, t):
            return secs
    total, found = 0.0, False
    for m in _PAIR_RE.finditer(t):
        raw, unit = m.group(1), m.group(2)
        mult = _unit_mult(unit)
        if not mult:
            continue
        n = float(raw.replace(",", ".")) if raw[0].isdigit() else float(NUM_WORDS[raw])
        total += n * mult
        found = True
    if not found:                       # bare unit: "a minute", greek "ωρα"
        for pat, mult in SEC_UNITS:
            if re.search(pat, t):
                return mult
    return int(total) if found and total > 0 else None


def parse_clock(text):
    if not text:
        return None
    t = strip_accents(text)
    m = re.search(r"(\d{1,2})[:.](\d{2})", t)
    if m:
        h, mi = int(m.group(1)), int(m.group(2))
    else:
        m = re.search(r"(\d{1,2})", t)
        if not m:
            for w, n in NUM_WORDS.items():
                if re.search(rf"\b{re.escape(w)}\b", t) and isinstance(n, int):
                    h, mi = n, 0
                    break
            else:
                if "μεσημερ" in t or "noon" in t:
                    return "12:00"
                if "μεσανυχτ" in t or "midnight" in t:
                    return "00:00"
                return None
        else:
            h, mi = int(m.group(1)), 0
    if re.search(r"\b(και )?μιση\b|half past", t):
        mi = 30
    if re.search(r"και τεταρτο|quarter past", t):
        mi = 15
    if re.search(r"παρα τεταρτο|quarter to", t):
        h, mi = (h - 1) % 24, 45
    # "7pm" and "7 pm" both count: no word boundary exists between a digit and a letter
    if re.search(r"(?<![a-z])pm\b|απογευμα|βραδυ|evening|night", t) and h < 12:
        h += 12
    if re.search(r"(?<![a-z])am\b|πρωι|morning", t) and h == 12:
        h = 0
    return f"{h % 24:02d}:{mi:02d}"


def parse_amount(text):
    if not text:
        return None
    m = re.search(r"-?\d+(?:[.,]\d+)?", text)
    if m:
        return float(m.group(0).replace(",", "."))
    t = strip_accents(text)
    for w, n in NUM_WORDS.items():
        if re.search(rf"\b{re.escape(w)}\b", t):
            return float(n)
    return None


def unit_of(text):
    if not text:
        return None
    t = strip_accents(text).strip()
    for w in re.findall(r"[a-zα-ω]+", t):
        if w in UNIT_ALIAS:
            return UNIT_ALIAS[w]
    return None


def calculate(expr):
    t = strip_accents(expr)
    t = re.sub(r"(?<=\d),(?=\d)", ".", t)            # 3,5 is a decimal in greek habit, not 35
    pct = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:percent|%|τοις εκατο)\s*(?:of|του|απο)?\s*(-?\d+(?:\.\d+)?)", t)
    if pct:
        return float(pct.group(1)) / 100 * float(pct.group(2))
    t = t.replace("^", "**")
    if re.search(r"\*\*\s*-?\d{3,}", t):                # 2^1000 is not a phone calculation
        return None
    for word, op in [("plus", "+"), ("συν", "+"), ("minus", "-"), ("πλην", "-"),
                     ("times", "*"), ("multiplied by", "*"), ("επι", "*"),
                     ("divided by", "/"), ("δια", "/"), ("over", "/"),
                     ("x", "*"), ("και", "+")]:
        t = re.sub(rf"\b{word}\b", op, t)
    if re.search(r"[a-zα-ω]", t):                       # a word survived: not arithmetic we understand
        return None
    t = re.sub(r"[^0-9\.\+\-\*/\(\)\s]", "", t)
    if not re.search(r"\d", t):
        return None
    try:
        return eval(compile(t, "<calc>", "eval"), {"__builtins__": {}}, {})
    except Exception:
        return None


def fmt_num(x):
    if x is None:
        return "?"
    return f"{x:.10g}" if isinstance(x, float) else str(x)


def fmt_secs(s):
    h, r = divmod(int(s), 3600)
    m, sec = divmod(r, 60)
    bits = [f"{h}h" if h else "", f"{m}m" if m else "", f"{sec}s" if sec else ""]
    return " ".join(b for b in bits if b) or "0s"


# ---------------------------------------------------------------- runtime
class Assistant:
    def __init__(self, speak=print):
        self.state = load_state()
        self.timers = {}
        self.speak = speak
        self.last = ""

    def _fire(self, name, label):
        self.timers.pop(name, None)
        self.speak(f"\n[TIMER] {label} is up.")

    # ---- individual skills -------------------------------------------------
    def timer_set(self, s):
        secs = parse_duration(s.get("duration", ""))
        if not secs:
            return "How long should the timer run for?"
        name = f"t{len(self.timers) + 1}"
        t = threading.Timer(secs, self._fire, args=(name, s.get("duration", "timer")))
        t.daemon = True
        t.start()
        self.timers[name] = (t, time.time() + secs, s.get("duration", ""))
        return f"Timer set for {fmt_secs(secs)}."

    def timer_cancel(self, s):
        if not self.timers:
            return "There is no timer running."
        for t, _, _ in self.timers.values():
            t.cancel()
        n = len(self.timers)
        self.timers.clear()
        return f"Cancelled {n} timer(s)."

    def timer_query(self, s):
        if not self.timers:
            return "No timer is running."
        parts = [fmt_secs(max(0, end - time.time())) for _, end, _ in self.timers.values()]
        return "Time left: " + ", ".join(parts)

    def alarm_set(self, s):
        clock = parse_clock(s.get("time", "")) or "?"
        date = s.get("date", "tomorrow")
        self.state["alarms"].append({"time": clock, "date": date})
        save_state(self.state)
        return f"Alarm set for {clock} ({date})."

    def alarm_cancel(self, s):
        n = len(self.state["alarms"])
        self.state["alarms"] = []
        save_state(self.state)
        return f"Cancelled {n} alarm(s)." if n else "There is no alarm set."

    def reminder_create(self, s):
        task = s.get("task") or s.get("content") or s.get("note")
        if not task:
            return "What should I remind you about?"
        r = {"task": task, "date": s.get("date", ""), "time": parse_clock(s.get("time", "")) or "",
             "created": datetime.now().isoformat(timespec="seconds")}
        self.state["reminders"].append(r)
        save_state(self.state)
        when = " ".join(x for x in [r["date"], r["time"]] if x)
        return f"Reminder saved: {task}" + (f" ({when})" if when else "")

    def reminder_list(self, s):
        rs = self.state["reminders"]
        if not rs:
            return "You have no reminders."
        return "Reminders:\n" + "\n".join(
            f"  {i+1}. {r['task']}" + (f"  [{r['date']} {r['time']}]".rstrip() if r["date"] or r["time"] else "")
            for i, r in enumerate(rs))

    def calendar_create(self, s):
        title = s.get("title") or s.get("content") or "event"
        e = {"title": title, "date": s.get("date", ""),
             "time": parse_clock(s.get("time", "")) or "", "person": s.get("person", "")}
        self.state["calendar"].append(e)
        save_state(self.state)
        return f"Added '{title}' " + " ".join(x for x in [e["date"], e["time"]] if x)

    def calendar_query(self, s):
        date = s.get("date")
        evs = [e for e in self.state["calendar"] if not date or e["date"] == date]
        if not evs:
            return f"Nothing in the calendar{(' for ' + date) if date else ''}."
        return "Calendar:\n" + "\n".join(
            f"  {e['time'] or '--:--'}  {e['title']}  {e['date']}" for e in evs)

    def note_create(self, s):
        body = s.get("note") or s.get("content") or ""
        if not body:
            return "What should I write down?"
        self.state["notes"].append({"text": body,
                                    "at": datetime.now().isoformat(timespec="seconds")})
        save_state(self.state)
        return f"Noted: {body}"

    def note_read(self, s):
        ns = self.state["notes"]
        if not ns:
            return "You have no notes."
        return "Notes:\n" + "\n".join(f"  {i+1}. {n['text']}" for i, n in enumerate(ns))

    def list_add(self, s):
        name = s.get("list_name", "shopping list")
        items = s.get("item", [])
        items = items if isinstance(items, list) else [items]
        items = [i for i in items if i]
        if not items:
            return "What should I add?"
        self.state["lists"].setdefault(name, []).extend(items)
        save_state(self.state)
        return f"Added {', '.join(items)} to the {name}."

    def list_read(self, s):
        name = s.get("list_name")
        if name and name in self.state["lists"]:
            items = self.state["lists"][name]
        else:
            items = [i for v in self.state["lists"].values() for i in v]
            name = name or "list"
        if not items:
            return f"The {name} is empty."
        return f"{name}: " + ", ".join(items)

    def math_calculate(self, s):
        expr = s.get("expression", "")
        v = calculate(expr)
        return f"{expr} = {fmt_num(v)}" if v is not None else "I could not work that out."

    def unit_convert(self, s):
        amount = parse_amount(s.get("amount", ""))
        a, b = unit_of(s.get("unit_from", "")), unit_of(s.get("unit_to", ""))
        if amount is None or not a or not b:
            return "I need an amount and both units."
        if (a, b) == ("c", "f"):
            return f"{fmt_num(amount)} C = {fmt_num(amount * 9 / 5 + 32)} F"
        if (a, b) == ("f", "c"):
            return f"{fmt_num(amount)} F = {fmt_num((amount - 32) * 5 / 9)} C"
        k = CONV.get((a, b))
        if not k:
            return f"I cannot convert {a} to {b} yet."
        return f"{fmt_num(amount)} {a} = {fmt_num(round(amount * k, 4))} {b}"

    def time_query(self, s):
        now = datetime.now()
        if s.get("location"):
            return f"[would call world clock] local time here is {now:%H:%M}"
        if s.get("date"):
            return f"Today is {now:%A %d %B %Y}."
        return f"It is {now:%H:%M} on {now:%A %d %B}."

    def volume_set(self, s):
        h = self.state["home"]
        lvl = parse_amount(s.get("level", ""))
        t = strip_accents(s["_text"])
        if lvl is None:
            if re.search(r"too loud|too high|πολυ δυνατα|down|quieter|lower|χαμηλ|μειωσ|σιγα", t):
                lvl = max(0, h["volume"] - 10)
            elif re.search(r"too quiet|too low|πολυ σιγα|up|louder|δυναμ|αυξησ|δυνατα", t):
                lvl = min(100, h["volume"] + 10)
            elif re.search(r"max|τερμα|μεγιστ", t):
                lvl = 100
            else:
                return "What volume?"
        h["volume"] = int(max(0, min(100, lvl)))
        save_state(self.state)
        return f"Volume {h['volume']}."

    def light_control(self, s):
        room = s.get("room", "all")
        t = strip_accents(s["_text"])
        on = not re.search(r"\b(off|kill|cut|out|shut|dim|σβησ|κλεισ|χαμηλωσ)\w*", t)
        self.state["home"]["lights"][room] = {
            "on": on, "color": s.get("color"), "brightness": s.get("brightness")}
        save_state(self.state)
        extra = ", ".join(f"{k} {v}" for k, v in
                          [("color", s.get("color")), ("brightness", s.get("brightness"))] if v)
        return f"Lights in {room}: {'on' if on else 'off'}" + (f" ({extra})" if extra else "")

    def device_control(self, s):
        dev = s.get("device", "device")
        t = strip_accents(s["_text"])
        if re.search(r"\bis the\b|ειναι ", t):
            st = self.state["home"]["devices"].get(dev)
            return f"The {dev} is {'on' if st else 'off'}."
        on = not re.search(r"\b(off|shut|kill|cut|stop|σβησ|κλεισ|σταματ)\w*", t)
        self.state["home"]["devices"][dev] = on
        save_state(self.state)
        return f"{dev} turned {'on' if on else 'off'}."

    def music_play(self, s):
        what = s.get("song") or s.get("playlist") or s.get("genre") or s.get("artist")
        by = f" by {s['artist']}" if s.get("song") and s.get("artist") else ""
        self.state["music"] = {"playing": f"{what}{by}", "paused": False}
        save_state(self.state)
        if not what:
            return "Play what?"
        _open("https://music.youtube.com/search?q=" + quote_plus(f"{what}{by}"))
        return f"Playing {what}{by} on YouTube Music."

    def music_control(self, s):
        t = strip_accents(s["_text"])
        m = self.state["music"]
        if re.search(r"next|skip|επομεν|αλλαξ", t):
            act = "next track"
        elif re.search(r"prev|back|προηγουμ", t):
            act = "previous track"
        elif re.search(r"pause|stop|σταματ|παυσ", t):
            m["paused"] = True
            act = "paused"
        elif re.search(r"resume|again|συνεχ|ξαναβαλ", t):
            m["paused"] = False
            act = "resumed"
        elif re.search(r"what song|τι τραγουδι", t):
            return f"Now playing: {m.get('playing') or 'nothing'}"
        else:
            act = "ok"
        save_state(self.state)
        return f"[player] {act}"

    # skills that live in the browser: the model supplies the arguments
    def weather_query(self, s):
        where, when = s.get("location", ""), s.get("date", "")
        _open("https://www.google.com/search?q=" + quote_plus(f"weather {where} {when}".strip()))
        return f"Opening the weather for {where or 'here'} {when}".rstrip() + "."

    def news_query(self, s):
        topic = s.get("topic", "")
        _open("https://news.google.com/search?q=" + quote_plus(topic) if topic
              else "https://news.google.com/")
        return f"Opening the news{(' on ' + topic) if topic else ''}."

    def search_web(self, s):
        q = s.get("query", "")
        if not q:
            return "Search for what?"
        _open("https://www.google.com/search?q=" + quote_plus(q))
        return f"Searching for {q}."

    def navigation_route(self, s):
        dest = s.get("location", "")
        if not dest:
            return "Where to?"
        _open("https://www.google.com/maps/dir/?api=1&destination=" + quote_plus(dest))
        return f"Opening directions to {dest}."

    def translate(self, s):
        phrase, lang = s.get("phrase", ""), s.get("language", "")
        code = LANG_CODES.get(strip_accents(lang), "en")
        _open(f"https://translate.google.com/?sl=auto&tl={code}&text={quote_plus(phrase)}&op=translate")
        return f"Translating '{phrase}' to {lang or 'english'}."

    # no phone is connected, so these only report what they understood
    def message_send(self, s):
        return (f"No phone connected. Understood: message to {s.get('person', '?')} "
                f"saying '{s.get('content', '')}'.")

    def call_make(self, s):
        return f"No phone connected. Understood: call {s.get('person', '?')}."

    # work domain: routing and judgement, never execution. Every request is
    # filed to data/work_queue.json with the slots the model extracted.
    def _file(self, kind, s, reply):
        q = self.state.setdefault("work_queue", [])
        q.append({"kind": kind, "text": s["_text"],
                  "slots": {k: v for k, v in s.items() if not k.startswith("_")},
                  "at": datetime.now().isoformat(timespec="minutes")})
        save_state(self.state)
        return reply

    def work_code(self, s):
        return self._file("code", s, f"Filed as a code task: {s.get('target') or s['_text']}. That needs a developer session, not me.")

    def work_review(self, s):
        return self._file("review", s, f"Filed a check on {s.get('target') or 'that'}. I will report what I find.")

    def work_status(self, s):
        q = self.state.get("work_queue", [])
        if not q:
            return "Nothing is queued. Everything I was given is done."
        return "Open items:\n" + "\n".join(f"  {i+1}. [{w['kind']}] {w['text']}" for i, w in enumerate(q[-8:]))

    def work_research(self, s):
        topic = s.get("topic") or s["_text"]
        _open("https://www.google.com/search?q=" + quote_plus(topic))
        return self._file("research", s, f"Researching {topic}: search first, then two or three sources, then an answer.")

    def work_write(self, s):
        who = f" to {s['person']}" if s.get("person") else ""
        return self._file("write", s, f"Drafting{who}: {s.get('topic') or s.get('content') or s['_text']}. Draft goes to the queue for your review.")

    def work_files(self, s):
        return self._file("files", s, f"File operation queued, not executed: {s['_text']}. Deleting needs your confirmation.")

    def work_ops(self, s):
        return self._file("ops", s, f"Ops action queued: {s['_text']}. I will confirm before anything touches a live system.")

    def work_business(self, s):
        return self._file("business", s, f"Business item queued: {s.get('topic') or s['_text']}.")

    def rule_set(self, s):
        rule = s.get("rule") or s.get("content") or s["_text"]
        rules = self.state.setdefault("rules", [])
        if rule not in rules:
            rules.append(rule)
        save_state(self.state)
        return f"Rule saved: {rule}. ({len(rules)} rules now.)"

    # conversational
    def greet(self, s):
        h = datetime.now().hour
        return "Good morning." if h < 12 else ("Good afternoon." if h < 18 else "Good evening.")

    def thanks(self, s):
        return "Any time."

    def bye(self, s):
        return "Talk later."

    def capabilities(self, s):
        return ("I handle timers, alarms, reminders, calendar, notes, lists, "
                "music, lights and devices, volume, maths, unit conversion, "
                "time, weather, news, search, navigation, translation, "
                "messages and calls. In English and Greek.")

    def repeat(self, s):
        return self.last or "I have not said anything yet."

    def cancel(self, s):
        return "Cancelled."

    HANDLERS = {
        "timer.set": timer_set, "timer.cancel": timer_cancel, "timer.query": timer_query,
        "alarm.set": alarm_set, "alarm.cancel": alarm_cancel,
        "reminder.create": reminder_create, "reminder.list": reminder_list,
        "calendar.create": calendar_create, "calendar.query": calendar_query,
        "note.create": note_create, "note.read": note_read,
        "list.add": list_add, "list.read": list_read,
        "math.calculate": math_calculate, "unit.convert": unit_convert,
        "time.query": time_query, "volume.set": volume_set,
        "light.control": light_control, "device.control": device_control,
        "music.play": music_play, "music.control": music_control,
        "weather.query": weather_query, "news.query": news_query,
        "search.web": search_web, "navigation.route": navigation_route,
        "translate": translate, "message.send": message_send, "call.make": call_make,
        "smalltalk.greet": greet, "smalltalk.thanks": thanks, "smalltalk.bye": bye,
        "assistant.capabilities": capabilities, "assistant.repeat": repeat,
        "assistant.cancel": cancel,
        "work.code": work_code, "work.review": work_review, "work.status": work_status,
        "work.research": work_research, "work.write": work_write, "work.files": work_files,
        "work.ops": work_ops, "work.business": work_business, "rule.set": rule_set,
    }

    def run(self, parsed):
        intent = parsed["intent"]
        if intent == "oos":
            return ("I did not understand that. It is outside what I was trained on."
                    if not parsed.get("below_threshold")
                    else "I am not sure what you mean.")
        fn = self.HANDLERS.get(intent)
        if not fn:
            return f"No handler for {intent}."
        slots = dict(parsed["slots"])
        slots["_text"] = parsed["text"]
        out = fn(self, slots)
        if intent != "assistant.repeat":
            self.last = out
        return out
