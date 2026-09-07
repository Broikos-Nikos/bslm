"""Stage 5: the trajectory generator. Real tool results, answers known in advance.

    python -m pretrain.wikidata                 (facts, songs, places; once)
    python -m pretrain.trajectories --facts 40000 --songs 4000 --local 16000 ...
        -> corpus/agent/{train,val}.txt  (one trajectory per block)
           corpus/agent/{train,val}.npy + _mask.npy  (tokens, 1 where the model speaks)
           corpus/agent/test.jsonl  (held out tasks with the truth, for the agent benchmark)

Every Result block is produced by running the action for real through
pretrain/agent_tools.Env (Wikipedia search and page extracts, YouTube result
pages, Open-Meteo forecasts, the local skills with a fresh state file). The
generator knows the truth (the Wikidata answer, the song and artist, the
calendar it wrote, the forecast it read), so it can write the Plan, Judge and
Deliver lines by rule, including the failures: a vague first query is really
run and really comes back without the answer, the judgement says why, and the
recovery is a real second call.

Protocol (plain text, the model owns Plan, Act, Judge, Ask, Deliver):

    Today is Monday 2026-09-07, 10:00. Home: Athens. Facts: rides a scooter.
    User: who directed inception
    Plan: A fact to check, not to guess: search for it.
    Act: search("Inception film director")
    Result:
    1. Inception: Inception is a 2010 science fiction action film written and directed by Christopher Nolan ...
    Judge: Result 1 states it: "directed by Christopher Nolan". That answers it.
    Deliver: Inception was directed by Christopher Nolan (Wikipedia: Inception).
"""
import argparse
import json
import os
import random
import re
import tempfile
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from bslm import gen_data as pools
from .agent_tools import Env, WEEKDAYS, fmt_results, fmt_videos, forecast, rain_hours, resolve_day, wiki_search, youtube_search

ROOT = Path(__file__).resolve().parent.parent
AGENT = ROOT / "corpus" / "agent"
EOT = "<|endoftext|>"
GEN_VERSION = "r9"      # bump whenever the text of a trajectory changes; the trajectory cache is keyed by it
MODEL_LINES = ("Plan:", "Act:", "Judge:", "Ask:", "Deliver:")

# ---------------------------------------------------------------- phrasing
Q = {   # question templates per relation; {s} is the subject
    "capital": ["what is the capital of {s}", "capital of {s}?", "whats the capital city of {s}",
                "which city is the capital of {s}", "tell me the capital of {s}"],
    "capital_region": ["what is the capital of {s}", "capital of {s}?", "which city is the capital of {s}"],
    "head_gov": ["who is the head of government of {s}", "who runs the government of {s}",
                 "who is the current head of government in {s}", "head of government of {s}?"],
    "head_state": ["who is the head of state of {s}", "who is the current head of state in {s}",
                   "head of state of {s}?"],
    "currency": ["what currency does {s} use", "currency of {s}?", "what money do they use in {s}",
                 "which currency is used in {s}"],
    "continent": ["which continent is {s} in", "what continent is {s} on", "{s} is on which continent"],
    "born_year": ["when was {s} born", "what year was {s} born", "{s} birth year?", "in which year was {s} born"],
    "died_year": ["when did {s} die", "what year did {s} die", "in which year did {s} die"],
    "birthplace": ["where was {s} born", "birthplace of {s}?", "in which city was {s} born"],
    "director": ["who directed {s}", "who is the director of {s}", "director of the film {s}?",
                 "who made the movie {s}"],
    "composer": ["who composed the music for {s}", "who wrote the score of {s}", "composer of the film {s}?"],
    "release_year": ["when was {s} released", "what year did {s} come out", "release year of the film {s}?",
                     "when did the movie {s} come out"],
    "author": ["who wrote {s}", "author of {s}?", "who is the author of the book {s}", "who wrote the book {s}"],
    "height": ["how tall is {s}", "how high is {s}", "height of {s} in meters?", "what is the elevation of {s}"],
    "founded": ["when was {s} founded", "what year was {s} founded", "founding year of {s}?",
                "when was the company {s} started"],
    "developer": ["who developed {s}", "who makes {s}", "developer of {s}?", "which company made {s}"],
    "country_of_city": ["which country is {s} in", "what country is {s} in", "{s} is in which country",
                        "where is {s}, which country"],
    "element_symbol": ["what is the chemical symbol of {s}", "symbol for {s}?", "chemical symbol for {s}"],
    "atomic_number": ["what is the atomic number of {s}", "atomic number of {s}?"],
    "sport": ["what sport does {s} play", "which sport is {s} known for", "{s} plays what sport"],
}
KEY = {   # the words the model adds to the subject to make a good query
    "capital": "capital", "capital_region": "capital", "head_gov": "head of government",
    "head_state": "head of state", "currency": "currency", "continent": "continent",
    "born_year": "born", "died_year": "died", "birthplace": "birthplace", "director": "film director",
    "composer": "film music composer", "release_year": "film release", "author": "book author",
    "height": "mountain elevation", "founded": "company founded", "developer": "developer",
    "country_of_city": "city country", "element_symbol": "chemical element symbol",
    "atomic_number": "atomic number", "sport": "athlete sport",
}
ALT = {   # a second, differently worded query for the retry
    "capital": "capital city", "capital_region": "seat of government", "head_gov": "prime minister",
    "head_state": "president", "currency": "money", "continent": "location", "born_year": "biography",
    "died_year": "death", "birthplace": "early life", "director": "directed by", "composer": "soundtrack",
    "release_year": "premiere", "author": "novel", "height": "summit metres", "founded": "history",
    "developer": "software", "country_of_city": "located in", "element_symbol": "element",
    "atomic_number": "periodic table", "sport": "career",
}
WHAT = {  # what is missing, for the judgement
    "capital": "the capital", "capital_region": "the capital", "head_gov": "who heads the government",
    "head_state": "who the head of state is", "currency": "the currency", "continent": "the continent",
    "born_year": "the birth year", "died_year": "the year of death", "birthplace": "the birthplace",
    "director": "the director", "composer": "the composer", "release_year": "the release year",
    "author": "the author", "height": "the height", "founded": "the founding year",
    "developer": "the developer", "country_of_city": "the country", "element_symbol": "the symbol",
    "atomic_number": "the atomic number", "sport": "the sport",
}
NOUN = dict(WHAT, head_gov="the head of government", head_state="the head of state")   # "so <noun> is <answer>"
SAY = {   # the delivered sentence, answer first; {s} subject, {a} answer
    "capital": "{a} is the capital of {s}", "capital_region": "{a} is the capital of {s}",
    "head_gov": "{a} is the head of government of {s}", "head_state": "{a} is the head of state of {s}",
    "currency": "The {a} is the currency of {s}", "continent": "{a} is the continent of {s}",
    "born_year": "{a} is the birth year of {s}", "died_year": "{a} is the year {s} died",
    "birthplace": "{a} is the birthplace of {s}", "director": "{a} directed {s}",
    "composer": "{a} composed the music for {s}", "release_year": "{a} is the release year of {s}",
    "author": "{a} wrote {s}", "height": "{a} metres is the height of {s}",
    "founded": "{a} is the founding year of {s}", "developer": "{a} developed {s}",
    "country_of_city": "{a} is the country of {s}", "element_symbol": "{a} is the symbol of {s}",
    "atomic_number": "{a} is the atomic number of {s}", "sport": "{a} is the sport of {s}",
}
CUE = {   # the words that announce the answer in a sentence; the quote starts there when they are close
    "capital": r"capital(?: city)?(?: is| of| was)?", "capital_region": r"capital(?: city)?(?: is| of| was)?",
    "head_gov": r"prime minister|head of government|chancellor|premier", "head_state": r"president|king|queen|monarch|head of state",
    "currency": r"currency|money", "continent": r"continent|located in", "born_year": r"born|birth",
    "died_year": r"died|death", "birthplace": r"born in|born at|native of|birthplace", "director": r"directed by|director",
    "composer": r"music by|composed by|score by|composer|soundtrack by", "release_year": r"released|premiered|release",
    "author": r"written by|author|novel by|book by|by", "height": r"elevation|height|metres|meters|m\b",
    "founded": r"founded|established|incorporated", "developer": r"developed by|developer|created by|made by",
    "country_of_city": r"city in|town in|located in|in the|in", "element_symbol": r"symbol", "atomic_number": r"atomic number",
    "sport": r"player|footballer|cyclist|swimmer|boxer|golfer|cricketer|racing driver|athlete|plays|competed",
}
KIND = {   # the kind of page each relation asks about, for titles that name several works
    "author": ("novel", "book"), "director": ("film",), "composer": ("film",), "release_year": ("film",),
    "developer": ("video game", "software"), "sport": (), "born_year": (), "died_year": (), "birthplace": (),
}


def page_for(items, s, rel):
    """The result that is the subject's own page: prefer a title whose
    qualifier matches the kind asked about ("Silence (novel)" for an author
    question), then the bare title, then any title containing it."""
    kinds = KIND.get(rel, ())
    ns = norm(s)
    ranked = []
    for i, x in enumerate(items):
        nt = norm(x["title"])
        if ns == nt:
            ranked.append((1, i))
        elif ns in nt or nt in ns:
            q = x["title"].lower()
            ranked.append((0 if any(k in q for k in kinds) else 2, i))
    if not ranked:
        return None, None
    ranked.sort()
    best = ranked[0][1]
    other = [i for _, i in ranked if i != best and norm(s) in norm(items[i]["title"])]
    return best, (other[0] if other and any(k in items[other[0]]["title"].lower() for k in ("film", "novel", "book", "album", "song", "game", "series")) else None)


PLAN_FACT = ["A fact to check, not to guess: search for it.",
             "I should not guess this. Search, then answer from the result.",
             "Fact question: search the name with the key word and read the results.",
             "Look it up rather than guess."]

SOUL = [("rides a scooter", "scooter"), ("drives a car", "car"), ("cycles to work", "bike"),
        ("takes the metro", "metro"), ("walks everywhere", "walk")]
HOMES = ["Athens", "Thessaloniki", "London", "Berlin", "Madrid", "Rome", "Lisbon", "Vienna", "Dublin", "Amsterdam"]


def norm(s):
    s = unicodedata.normalize("NFD", str(s).lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9 ]+", " ", s).strip()


def found_in(answer, text):
    """Is the answer stated in the text: full label, a whole number, or the surname."""
    n_ans, n_txt = norm(answer), " " + norm(text) + " "
    if not n_ans:
        return False
    if re.fullmatch(r"\d+(\.\d+)?", n_ans):
        return re.search(r"(?<![\d.])" + re.escape(n_ans.split(".")[0]) + r"(?![\d])", n_txt.replace(",", "")) is not None
    if " " + n_ans + " " in n_txt:
        return True
    parts = n_ans.split()
    if len(parts) >= 2 and len(parts[-1]) >= 4 and " " + parts[-1] + " " in n_txt and parts[0][:3] in n_txt:
        return True
    return False


def fragment(answer, text, width=60, rel=None):
    """A short quote around the answer, for the judgement line. When the
    relation's cue words ("directed by", "capital") sit shortly before the
    answer, the quote starts at the cue, so the model learns to read the
    words that announce the answer instead of any name nearby."""
    t = re.sub(r"\s+", " ", text)
    i = norm(t).find(norm(answer).split()[-1]) if norm(answer) else -1
    if i < 0:
        return t[:width]
    lo = max(0, i - width // 2)
    if rel in CUE:
        before = t[max(0, i - 45):i]
        m = None
        for m in re.finditer(CUE[rel], before, re.I):
            pass
        if m:
            lo = max(0, i - 45) + m.start()
            hi = i + len(answer) + 12
            return (t[lo:hi].rsplit(" ", 1)[0] if hi < len(t) else t[lo:]).strip(" ,;:")
    out = t[lo:lo + width]
    if lo > 0:
        out = out.split(" ", 1)[-1]          # start on a word boundary
    if lo + width < len(t):
        out = out.rsplit(" ", 1)[0]          # end on one too
    return out.strip(" ,;:")


def misspell(s, rng):
    w = [x for x in s.split() if len(x) > 4]
    if not w:
        return s + "x"
    x = rng.choice(w)
    i = rng.randrange(1, len(x) - 1)
    return s.replace(x, x[:i] + x[i + 1:], 1)


# ---------------------------------------------------------------- rendering
class Traj:
    """Collects the lines and which of them the model owns."""

    def __init__(self, header, user):
        self.lines = [(header, 0), ("User: " + user, 0)]
        self.acts = []

    def model(self, tag, text):
        self.lines.append((f"{tag}: {text}", 1))

    def result(self, text):
        self.lines.append(("Result:\n" + text.strip(), 0))

    def user(self, text):
        self.lines.append(("User: " + text, 0))

    def act(self, env, line):
        self.model("Act", line)
        r = env.act(line)
        self.acts.append((line, r))
        self.result(r)
        return r

    def text(self):
        return "\n".join(t for t, _ in self.lines) + "\n"

    def spans(self):
        """Character spans owned by the model, over text()."""
        out, pos = [], 0
        for t, m in self.lines:
            if m:
                out.append((pos, pos + len(t) + 1))
            pos += len(t) + 1
        return out


def header(now, home, soul, tools=None, earlier=None):
    """The context the model sees before the user's line: date and soul, the
    tools the wrapper registered, and one compressed line per earlier turn
    of this round (no result text, ever: the model must not quote stale
    results)."""
    h = f"Today is {now.strftime('%A %Y-%m-%d, %H:%M')}. Home: {home}. Facts: {soul}."
    if tools:
        h += "\nTools: " + "; ".join(f"{n} ({d})" for n, d in tools)
    for u, act, said in (earlier or []):
        h += f'\nEarlier: "{u}" -> {act or "no action"} -> "{said[:90]}"'
    return h


def new_env_with(now, library=None, tools=None):
    env, st = new_env(now, library)
    env.tools = list(tools or [])
    return env, st


TOOLS = [   # (name, what it does, the verbs people use for it); a wrapper registers real ones the same way
    ("qr file receiver", "receives files sent by QR code", ["open", "start", "launch"]),
    ("obs", "records the screen", ["start recording with", "stop recording with", "open"]),
    ("vpn", "the VPN connection", ["connect", "disconnect", "turn on", "turn off"]),
    ("backup", "backs up the documents folder", ["run", "start"]),
    ("scanner", "scans a document to PDF", ["open", "start", "run"]),
    ("print queue", "shows the printer queue", ["open", "show", "check"]),
    ("music server", "the home music server", ["restart", "stop", "start"]),
    ("zorbo", "syncs the photos to the NAS", ["run", "start", "open"]),
    ("mailbot", "checks the inbox and reads new mail", ["run", "open", "start"]),
    ("ticker", "shows the stock ticker", ["open", "close", "show"]),
    ("garage", "the garage door", ["open", "close"]),
    ("nightwatch", "arms the cameras", ["arm", "disarm", "open"]),
    ("kiosk", "the shop kiosk screen", ["open", "close", "restart"]),
    ("timelapse", "the timelapse camera", ["start", "stop"]),
]


def rand_tools(rng, k=None):
    k = rng.choice([0, 0, 1, 2, 3, 4]) if k is None else k
    return [(n, d) for n, d, _ in rng.sample(TOOLS, k)]


def new_env(now, library=None):
    f = tempfile.NamedTemporaryFile(prefix="bslm_traj_", suffix=".json", delete=False)
    f.close()
    os.remove(f.name)
    return Env(f.name, now=now, library=library), f.name


def drop_state(path):
    try:
        os.remove(path)
    except OSError:
        pass


# ---------------------------------------------------------------- families
def fact_traj(fact, rng, now, home, soul, shape):
    """shape: 'direct' (good query first), 'bad_first' (vague query, then good),
    'open' (good query, answer only on the page). Reality decides the rest."""
    rel, s, a = fact["rel"], fact["subject"], fact["answer"]
    q_user = rng.choice(Q[rel]).format(s=s)
    canonical = s
    if rng.random() < 0.3:
        q_user = q_user.lower()
        s = s.lower()          # the model copies what the user typed, it does not re-case names from memory
    t = Traj(header(now, home, soul), q_user)
    env, st = new_env(now)
    t.model("Plan", rng.choice(PLAN_FACT))
    good = f"{s} {KEY[rel]}"
    queries = []
    if shape == "bad_first":
        bad = rng.choice([KEY[rel], f"{misspell(s, rng)} {KEY[rel]}", f"{s} {ALT[rel]}"[:40]])
        queries.append(bad)
    queries.append(good)
    truth = {"family": "fact", "rel": rel, "subject": canonical, "answer": a, "aliases": fact.get("aliases", []), "user": q_user}
    cands = [a] + [x for x in fact.get("aliases", []) if x]

    def where(text):
        return next((c for c in cands if found_in(c, text)), None)

    def said(s_, a_):
        if rel == "sport" and a_ != a:
            return f"{s_} is a {a_}" if not a_.endswith("ing") else f"{s_} does {a_}"
        return SAY[rel].format(s=s_, a=a_)
    for qi, q in enumerate(queries):
        if shape == "bad_first" and qi == 0:
            # the vague query is served by DBpedia Lookup: real, noisy, and it
            # spares Wikipedia's rate limit for the query that has to succeed
            env.next_backend = "lookup"
        r = t.act(env, f'search("{q}")')
        items = env.last_results
        if items and "error" in items[0]:
            drop_state(st)
            return None, None
        hit = next((i for i, x in enumerate(items) if where(x["snippet"])), None)
        if hit is not None and shape != "open":
            a_found = where(items[hit]["snippet"])
            # name the answer in the judgement, so the delivery copies it from
            # the line above instead of hunting through the result block; and
            # say out loud when the first result is about something else
            skip = ""
            if hit > 0 and norm(s) not in norm(items[0]["title"]):
                skip = f"Result 1 is about {items[0]['title']}, not {s}. "
            elif hit > 0 and rel == "author" and "film" in items[0]["title"].lower():
                skip = "Result 1 is the film, not the book. "
            elif hit > 0 and rel in ("director", "composer") and re.search(r"novel|book|\(album\)|\(song\)", items[0]["title"].lower()):
                skip = "Result 1 is not the film. "
            t.model("Judge", f'{skip}Result {hit + 1} states it: "{fragment(a_found, items[hit]["snippet"], rel=rel)}", so {NOUN[rel]} is {a_found}. That answers it.')
            t.model("Deliver", said(s, a_found) + f" (Wikipedia: {items[hit]['title']}).")
            drop_state(st)
            return t, truth
        # not in the snippets (or we want the page): is the subject's page here?
        page_i, other_i = page_for(items, s, rel)
        if page_i is not None:
            which = ""
            if other_i is not None:
                which = f" Result {other_i + 1} is {items[other_i]['title']}, a different work with the same name;"
            if hit is not None:
                t.model("Judge", f"Result {page_i + 1} is the page for {s};{which} open it to be sure.")
            else:
                t.model("Judge", f"The results mention {s} but none states {WHAT[rel]}.{which} Open result {page_i + 1}, the page itself.")
            page = t.act(env, f"open({page_i + 1})")
            a_found = where(page)
            if a_found:
                t.model("Judge", f'The page says: "{fragment(a_found, page, rel=rel)}", so {NOUN[rel]} is {a_found}. That answers it.')
                t.model("Deliver", said(s, a_found) + f" (Wikipedia: {items[page_i]['title']}).")
                drop_state(st)
                return t, truth
            if qi + 1 < len(queries):
                t.model("Judge", f"The page opening does not state {WHAT[rel]}. Try a query with the key word.")
                continue
            alt = f"{s} {ALT[rel]}"
            t.model("Judge", f"The page opening does not state {WHAT[rel]}. One more query, worded differently.")
            r2 = t.act(env, f'search("{alt}")')
            items2 = env.last_results
            hit2 = next((i for i, x in enumerate(items2) if where(x["snippet"])), None) if items2 and "error" not in items2[0] else None
            if hit2 is not None:
                a_found = where(items2[hit2]["snippet"])
                t.model("Judge", f'Result {hit2 + 1} states it: "{fragment(a_found, items2[hit2]["snippet"], rel=rel)}", so {NOUN[rel]} is {a_found}. That answers it.')
                t.model("Deliver", said(s, a_found) + f" (Wikipedia: {items2[hit2]['title']}).")
                drop_state(st)
                return t, truth
            t.model("Judge", f"Still not stated. Two searches and the page did not give {WHAT[rel]}; say what I found and stop.")
            t.model("Deliver", f"I could not confirm {WHAT[rel]} of {s}. The closest source I found is the Wikipedia page \"{items[page_i]['title']}\", which does not state it.")
            truth["answer"] = None
            drop_state(st)
            return t, truth
        # no page for the subject in this list
        if qi + 1 < len(queries):
            if not items:
                t.model("Judge", "No results at all; the query was too vague. Search with the full name and the key word.")
            else:
                t.model("Judge", f"Nothing here is about {s}: the query was off. Search with the full name and the key word.")
            continue
        t.model("Judge", f"The results are not about {s} and none states {WHAT[rel]}. I cannot find it; say so.")
        t.model("Deliver", f"I could not find {WHAT[rel]} of {s}; the search did not return a page about it.")
        truth["answer"] = None
        drop_state(st)
        return t, truth
    drop_state(st)
    return None, None


SONG_ASK = ["play {t}", "play {t} by {a}", "put on {t}", "i want to hear {t} by {a}", "can you play {t}",
            "play the song {t}", "play {t} from {a}", "put on {t} by {a}"]


def pick_video(items, title, artist):
    """The first result whose title carries the song title and, when an
    artist is known, the artist too; artist None means the user named none."""
    nt, na = norm(title), (norm(artist) if artist else "")
    for i, v in enumerate(items):
        vt = norm(v["title"]) + " " + norm(v.get("channel", ""))
        if nt and nt in vt and (not na or na in vt or na.split()[0] in vt):
            return i
    return None


def song_traj(song, rng, now, home, soul, library, in_library):
    title, artist = song["title"], song["artist"]
    if in_library:
        library = library + [f"{title} - {artist}"]
    user = rng.choice(SONG_ASK).format(t=title, a=artist)
    canonical_title, canonical_artist = title, artist
    if rng.random() < 0.4:
        user = user.lower()
        title, artist = title.lower(), artist.lower()     # copy the user's words, do not re-case from memory
    t = Traj(header(now, home, soul), user)
    env, st = new_env(now, library=library)
    truth = {"family": "song", "title": canonical_title, "artist": canonical_artist, "user": user, "in_library": in_library}
    t.model("Plan", rng.choice(["Check the music library first; YouTube if it is not there.",
                                "Library first, then YouTube, and verify the title before playing.",
                                "Look in the library, fall back to YouTube, pick the matching title."]))
    r = t.act(env, f'library("{title}")')
    hits = env.last_library
    k = next((i for i, h in enumerate(hits) if norm(title) in norm(h)), None)
    if k is not None:
        t.model("Judge", f"Result {k + 1} is the song, in the library.")
        t.act(env, f"play({k + 1})")
        t.model("Deliver", f"Playing {title} by {artist} from your library.")
        truth["expected"] = "library"
        drop_state(st)
        return t, truth
    known = norm(artist) in norm(user)          # did the user name the artist at all
    if known:
        t.model("Judge", "Not in the library. Search YouTube with the title and the artist.")
        r = t.act(env, f'youtube("{title} {artist}")')
    else:
        t.model("Judge", "Not in the library. No artist was given, so search YouTube with the title alone.")
        r = t.act(env, f'youtube("{title}")')
    artist_q = artist if known else None
    vids = env.last_videos
    if not vids or "error" in vids[0]:
        drop_state(st)
        return None, None
    k = pick_video(vids, title, artist_q)
    if k is None:
        if known:
            t.model("Judge", "None of these is the song itself: the titles do not match. Search once more with the artist first.")
            r = t.act(env, f'youtube("{artist} {title} official")')
        else:
            t.model("Judge", "None of these is the song itself: the titles do not match. Search once more with the word song added.")
            r = t.act(env, f'youtube("{title} song")')
        vids = env.last_videos
        if not vids or "error" in vids[0]:
            drop_state(st)
            return None, None
        k = pick_video(vids, title, artist_q)
        if k is None:
            t.model("Judge", "Still no result whose title matches the song. Do not play a random video; say so.")
            t.model("Deliver", f"I could not find {title} by {artist} on YouTube; the closest result is \"{vids[0]['title']}\". Want that one?")
            truth["expected"] = "none"
            drop_state(st)
            return t, truth
    t.model("Judge", f'Result {k + 1} matches the title{" and the artist" if known else ""}: "{vids[k]["title"]}".')
    t.act(env, f"play({k + 1})")
    t.model("Deliver", f"Playing \"{vids[k]['title']}\" on YouTube.")
    truth["expected"] = "youtube"
    truth["video"] = vids[k]["title"]
    drop_state(st)
    return t, truth


# local skills: (user templates, action builder, deliver builder)
def P(name):
    v = getattr(pools, name)
    return v["en"] if isinstance(v, dict) else v


def local_cases(rng, now):
    """One random local task: returns dict(user, plan, act, deliver(result), setup=[acts before], truth)."""
    kind = rng.choice(["timer", "timer", "timer_cancel", "timer_left", "alarm", "alarm", "remind", "remind",
                       "reminders", "list_add", "list_add", "list_read", "note", "notes", "event", "event",
                       "agenda", "agenda", "calc", "calc", "convert", "convert", "time", "lights", "lights",
                       "lights_level", "lights_level", "lights_scene", "plug", "plug", "plug", "device", "volume",
                       "tool", "tool", "tool", "weather", "weather"])
    d = {"kind": kind, "setup": [], "turns": [], "tools": rand_tools(rng)}
    if kind == "timer":
        dur = rng.choice(pools.durations("en"))
        d["user"] = rng.choice(["set a timer for {d}", "timer {d}", "start a {d} timer", "remind me in {d}",
                                "countdown {d} please", "put a timer on for {d}"]).format(d=dur)
        d["plan"] = "A timer: set it and confirm."
        d["act"] = f'timer("{dur}")'
        d["deliver"] = lambda r: r if r.startswith("Timer set") else f"I could not set that timer: {r}"
    elif kind == "timer_cancel":
        running = rng.random() < 0.5
        if running:
            d["setup"].append(f'timer("{rng.choice(pools.durations("en"))}")')
        d["user"] = rng.choice(["cancel the timer", "stop the timer", "kill the timer", "cancel my countdown"])
        d["plan"] = "Cancel the timer if one is running."
        d["act"] = "cancel_timer()"
        d["deliver"] = lambda r: "Timer cancelled." if r.startswith("Cancelled") else "There is no timer running."
    elif kind == "timer_left":
        running = rng.random() < 0.6
        if running:
            d["setup"].append(f'timer("{rng.choice(pools.durations("en"))}")')
        d["user"] = rng.choice(["how much time is left on the timer", "how long until the timer ends", "timer status"])
        d["plan"] = "Ask the timer for the remaining time."
        d["act"] = "timer_left()"
        d["deliver"] = lambda r: r
    elif kind == "alarm":
        tm, dt = rng.choice(pools.times("en")), rng.choice(["tomorrow", "tomorrow", "monday", "friday", "today", "saturday"])
        d["user"] = rng.choice(["set an alarm for {t} {d}", "wake me up at {t} {d}", "alarm {t} {d}",
                                "set my alarm to {t} {d}", "wake me at {t} {d}"]).format(t=tm, d=dt)
        d["plan"] = "An alarm: set it for that time and day."
        d["act"] = f'alarm("{tm}", "{dt}")'
        d["deliver"] = lambda r: r
    elif kind == "remind":
        task = rng.choice(P("TASK"))
        dt = rng.choice(["tomorrow", "friday", "monday", "today", "tonight", "next week", "saturday"])
        tm = rng.choice(pools.times("en")) if rng.random() < 0.6 else ""
        when = f"{dt} at {tm}" if tm else dt
        d["user"] = rng.choice(["remind me to {x} {w}", "reminder: {x} {w}", "dont let me forget to {x} {w}",
                                "set a reminder to {x} {w}", "remind me {w} to {x}"]).format(x=task, w=when)
        d["plan"] = "A reminder: save the task with the day and time."
        d["act"] = f'remind("{task}", "{dt}", "{tm}")'
        d["deliver"] = lambda r, task=task, when=when: f"I will remind you to {task} {when}." if r.startswith("Reminder") else r
    elif kind == "reminders":
        for _ in range(rng.randrange(0, 4)):
            d["setup"].append(f'remind("{rng.choice(P("TASK"))}", "{rng.choice(["tomorrow", "friday", "monday"])}", "")')
        d["user"] = rng.choice(["what are my reminders", "list my reminders", "show reminders", "anything i have to do"])
        d["plan"] = "Read the saved reminders."
        d["act"] = "reminders()"
        d["deliver"] = lambda r: ("You have no reminders." if "no reminders" in r.lower() or not r.strip() else
                                  "Your reminders: " + "; ".join(re.sub(r"^\s*\d+\.\s*", "", x).strip() for x in r.splitlines()[1:]) + ".")
    elif kind == "list_add":
        items = rng.sample(P("ITEM"), rng.randrange(1, 4))
        lists = ["shopping", "hardware", "packing"]
        ambiguous = rng.random() < 0.25
        if ambiguous:
            d["setup"] += [f'list_add("{lists[0]}", ["{rng.choice(P("ITEM"))}"])', f'list_add("{lists[1]}", ["screws"])']
            d["user"] = rng.choice(["add {i} to the list", "put {i} on my list", "add {i}"]).format(i=", ".join(items))
            d["plan"] = "Add items to a list; the user has more than one list, so ask which."
            d["ask"] = ("Which list, shopping or hardware?", rng.choice(["shopping", "the shopping one", "hardware", "hardware list"]))
            chosen = "hardware" if "hardware" in d["ask"][1] else "shopping"
            d["act"] = f'list_add("{chosen}", {json.dumps(items)})'
            d["deliver"] = lambda r, items=items, chosen=chosen: f"Added {', '.join(items)} to the {chosen} list."
        else:
            ln = rng.choice(["shopping", "shopping", "grocery", "todo", "packing"])
            d["user"] = rng.choice(["add {i} to the {l} list", "put {i} on the {l} list", "{l} list: add {i}",
                                    "add {i} to my {l} list"]).format(i=", ".join(items), l=ln)
            d["plan"] = "Add the items to that list."
            d["act"] = f'list_add("{ln}", {json.dumps(items)})'
            d["deliver"] = lambda r, items=items, ln=ln: f"Added {', '.join(items)} to the {ln} list."
    elif kind == "list_read":
        ln = rng.choice(["shopping", "grocery", "todo", "packing"])
        empty = rng.random() < 0.3
        if not empty:
            d["setup"].append(f'list_add("{ln}", {json.dumps(rng.sample(P("ITEM"), rng.randrange(1, 5)))})')
        d["user"] = rng.choice(["whats on my {l} list", "read the {l} list", "show my {l} list", "{l} list?"]).format(l=ln)
        d["plan"] = "Read that list."
        d["act"] = f'list_read("{ln}")'
        d["deliver"] = lambda r, ln=ln: (f"Your {ln} list is empty." if "empty" in r.lower() or "no " in r.lower()[:6] else
                                          f"Your {ln} list: " + r.split(":", 1)[-1].strip() + ".")
    elif kind == "note":
        note = rng.choice(P("NOTE"))
        d["user"] = rng.choice(["note that {n}", "take a note: {n}", "remember that {n}", "write down {n}", "note: {n}"]).format(n=note)
        d["plan"] = "Save the note."
        d["act"] = f'note("{note}")'
        d["deliver"] = lambda r, note=note: f"Noted: {note}."
    elif kind == "notes":
        for _ in range(rng.randrange(0, 3)):
            d["setup"].append(f'note("{rng.choice(P("NOTE"))}")')
        d["user"] = rng.choice(["read my notes", "what notes do i have", "show my notes", "my notes?"])
        d["plan"] = "Read the saved notes."
        d["act"] = "notes()"
        d["deliver"] = lambda r: ("You have no notes." if "no notes" in r.lower() else
                                  "Your notes: " + "; ".join(re.sub(r"^\s*\d+\.\s*", "", x).strip() for x in r.splitlines()[1:]) + ".")
    elif kind == "event":
        ev, dt, tm = rng.choice(P("EVENT")), rng.choice(["tomorrow", "monday", "friday", "today", "thursday", "next tuesday"]), rng.choice(pools.times("en"))
        d["user"] = rng.choice(["add {e} to my calendar {d} at {t}", "schedule {e} {d} at {t}", "put {e} in the calendar for {d} {t}",
                                "new event: {e}, {d} {t}", "book {e} for {d} at {t}"]).format(e=ev, d=dt, t=tm)
        d["plan"] = "A calendar entry: add it with the day and time."
        d["act"] = f'event("{ev}", "{dt}", "{tm}")'
        d["deliver"] = lambda r, ev=ev, dt=dt, tm=tm: f"Added {ev} on {dt} at {tm}." if r.startswith("Added") else r
    elif kind == "agenda":
        dt = rng.choice(["today", "tomorrow", "monday", "friday", "wednesday"])
        n = rng.choice([0, 1, 1, 2, 3])
        for _ in range(n):
            d["setup"].append(f'event("{rng.choice(P("EVENT"))}", "{dt}", "{rng.choice(pools.times("en"))}")')
        d["user"] = rng.choice(["what do i have {d}", "whats on my calendar {d}", "am i free {d}", "my schedule for {d}?",
                                "any appointments {d}"]).format(d=dt)
        d["plan"] = "Read the calendar for that day."
        d["act"] = f'agenda("{dt}")'
        d["deliver"] = lambda r, dt=dt: (f"Nothing in the calendar for {dt}, you are free." if "Nothing" in r else
                                          f"On {dt}: " + "; ".join(re.sub(r"\s+" + re.escape(dt) + r"\s*$", "", re.sub(r"\s+", " ", x).strip()) for x in r.splitlines()[1:]) + ".")
    elif kind == "calc":
        expr = rng.choice(pools.MATH)
        d["user"] = rng.choice(["whats {e}", "calculate {e}", "{e} = ?", "how much is {e}", "compute {e}"]).format(e=expr)
        d["plan"] = "Arithmetic: compute it, do not guess."
        d["act"] = f'calc("{expr}")'
        d["deliver"] = lambda r: (r.replace(" = ", " equals ") + "." if " = " in r else f"I could not compute that: {r}")
    elif kind == "convert":
        fu, tu = rng.choice(pools.UNIT_PAIRS)
        amt = rng.choice([1, 2, 5, 10, 12, 25, 50, 100, 250, 3.5, 7.5])
        d["user"] = rng.choice(["convert {a} {f} to {t}", "how many {t} is {a} {f}", "{a} {f} in {t}", "{a} {f} to {t} please"]).format(a=amt, f=fu, t=tu)
        d["plan"] = "A unit conversion: compute it."
        d["act"] = f'convert({amt}, "{fu}", "{tu}")'
        d["deliver"] = lambda r: (r + "." if " = " in r else f"I could not convert that: {r}")
    elif kind == "time":
        d["user"] = rng.choice(["what time is it", "whats the time", "time?", "what day is it today", "whats the date"])
        d["plan"] = "Read the clock."
        d["act"] = "time()"
        d["deliver"] = lambda r: r
    elif kind == "lights":
        st = rng.choice(["on", "off"])
        room = rng.choice(P("ROOM"))
        color = rng.choice(P("COLOR")) if st == "on" and rng.random() < 0.3 else None
        verb = {"on": rng.choice(["turn on", "switch on", "lights on in", "put on"]), "off": rng.choice(["turn off", "switch off", "kill", "lights off in"])}[st]
        d["user"] = f"{verb} the {room} lights" if "in" not in verb else f"{verb} the {room}"
        if color:
            d["user"] = rng.choice([f"make the {room} lights {color}", f"set the {room} lights to {color}"])
        d["plan"] = "Home control: set the lights."
        d["act"] = f'lights("{st}", "{room}"' + (f', "{color}")' if color else ")")
        d["deliver"] = lambda r: r + "."
    elif kind == "tool":
        # a tool from the header's list, or one that is not there (a third of the time)
        listed = [t for t in TOOLS if t[0] in {n for n, _ in d["tools"]}]
        if not listed or rng.random() < 0.3:
            name, _, verbs = rng.choice([t for t in TOOLS if t[0] not in {n for n, _ in d["tools"]}] or TOOLS)
            verb = rng.choice(verbs)
            d["user"] = rng.choice(["{v} the {n}", "{v} {n}", "can you {v} the {n}", "{v} the {n} please"]).format(v=verb, n=name)
            names = ", ".join(n for n, _ in d["tools"])
            d["plan"] = f"No tool called {name} in my list; say what I have instead of pretending." if d["tools"] else \
                "No tools are registered on this device; say so."
            d["act"] = None
            d["deliver"] = (lambda r, name=name, names=names: f"I have no tool called {name}. The tools I have: {names}.") if d["tools"] else \
                (lambda r, name=name: f"I have no tool called {name}, and no tools are registered on this device.")
        else:
            name, _, verbs = rng.choice(listed)
            verb = rng.choice(verbs)
            d["user"] = rng.choice(["{v} the {n}", "{v} {n}", "can you {v} the {n}", "{v} the {n} please", "launch {n}" if verb == "open" else "{v} {n} now"]).format(v=verb, n=name)
            d["plan"] = f"{name} is in my tool list: use it."
            d["act"] = f'tool("{name}", "{verb}")'
            d["deliver"] = lambda r, name=name, verb=verb: f"Done: {verb} {name}." if "done" in r else r
    elif kind == "weather":
        place = rng.choice(HOMES + ["London", "Paris", "Berlin", "New York", "Tokyo", "Rome"])
        day = rng.choice(["today", "tomorrow", "tomorrow", "the day after tomorrow", "friday", "saturday"])
        d["user"] = rng.choice(["whats the weather {d} in {p}", "weather {d} in {p}", "is it going to rain {d} in {p}",
                                "do i need an umbrella {d} in {p}", "how hot is it {d} in {p}", "{p} weather {d}"]).format(d=day, p=place)
        d["plan"] = "A forecast question: read the forecast for that place and day."
        d["act"] = f'weather("{place}", "{day}")'
        d["deliver"] = lambda r, place=place, day=day: weather_summary(r, place, day)
    elif kind == "lights_level":
        room = rng.choice(P("ROOM"))
        lv = rng.choice(["full", "high", "half", "soft", "low", "dim", "25", "50", "75", "100", "10"])
        pct = f"{lv} percent" if lv.isdigit() else lv
        d["user"] = rng.choice(["{r} lights {l}", "set the {r} lights to {l}", "{l} lights in the {r}", "make the {r} lights {l}",
                                "put the {r} lights on {l}", "{r} lights at {l}", "turn the {r} lights to {l}",
                                "i want {l} light in the {r}"]).format(r=room, l=pct)
        if lv in ("dim", "low", "soft") and rng.random() < 0.5:
            d["user"] = rng.choice(["dim the {r} lights", "lower the {r} lights", "make the {r} lights softer"]).format(r=room)
            lv = "soft" if "soft" in d["user"] else lv
        d["plan"] = "Home control: set the lights to that level."
        d["act"] = f'lights("{lv}", "{room}")'
        d["deliver"] = lambda r: r + "."
    elif kind == "lights_scene":
        room = rng.choice(P("ROOM"))
        sc = rng.choice(["movie", "reading", "cozy", "relax", "party", "dinner", "focus", "sleep"])
        d["user"] = rng.choice(["{s} mode in the {r}", "set the {r} lights to {s}", "{r} lights {s} mode", "put the {r} in {s} mode",
                                "{s} lights in the {r}", "switch the {r} lights to the {s} scene"]).format(s=sc, r=room)
        d["plan"] = "Home control: a lighting scene, the preset under that name."
        d["act"] = f'lights("{sc}", "{room}")'
        d["deliver"] = lambda r: r + "."
    elif kind == "plug":
        dev = rng.choice(["water heater", "boiler", "coffee machine", "fan", "space heater", "christmas lights",
                          "garden lights", "pool pump", "kettle", "air purifier", "dehumidifier", "printer"])
        on = rng.random() < 0.65
        verb = rng.choice(["open", "turn on", "switch on", "start", "power on", "put on"] if on else
                          ["close", "turn off", "switch off", "stop", "power off", "shut off", "kill"])
        d["user"] = rng.choice(["{v} the {d}", "{v} {d}", "can you {v} the {d}", "{v} the {d} please",
                                "{d} {s}"]).format(v=verb, d=dev, s="on" if on else "off")
        if dev in ("water heater", "boiler") and on and rng.random() < 0.4:
            d["user"] = rng.choice(["open the {d} for a bath", "i want a warm shower, {v} the {d}", "heat the water",
                                    "{v} the {d}, i want to shower in an hour"]).format(d=dev, v=verb)
        d["plan"] = "Home control: a switched appliance, on or off."
        d["act"] = f'switch("{dev}", "{"on" if on else "off"}")'
        d["deliver"] = lambda r: r if r.endswith(".") else r + "."
    elif kind == "device":
        dev, st = rng.choice(P("DEVICE")), rng.choice(["on", "off"])
        d["user"] = rng.choice(["turn {s} the {d}", "switch {s} the {d}", "{d} {s} please", "can you turn the {d} {s}"]).format(s=st, d=dev)
        d["plan"] = "Home control: switch the device."
        d["act"] = f'device("{dev}", "{st}")'
        d["deliver"] = lambda r: r if r.endswith(".") else r + "."
    else:  # volume
        lvl = rng.choice(pools.LEVEL[:21])
        d["user"] = rng.choice(["set the volume to {l}", "volume {l}", "volume to {l} percent", "put the volume at {l}"]).format(l=lvl)
        d["plan"] = "Set the volume."
        d["act"] = f"volume({lvl})"
        d["deliver"] = lambda r: r
    return d


def weather_summary(r, place, day):
    if r.startswith("no forecast"):
        return f"I could not get a forecast for {place}: {r.split(':', 1)[-1].strip()}."
    temps = [int(x) for x in re.findall(r"(-?\d+)C", r)]
    rain = re.findall(r"(\d\d:\d\d) -?\d+C (\d+)%", r)
    wet = [(h, int(p)) for h, p in rain if int(p) >= 40]
    lo, hi = (min(temps), max(temps)) if temps else ("?", "?")
    if wet:
        h, p = max(wet, key=lambda x: x[1])
        return f"{day.capitalize()} in {place}: {lo} to {hi}C, rain likely, up to {p}% around {h}. Take an umbrella."
    return f"{day.capitalize()} in {place}: {lo} to {hi}C, no rain expected."


def local_traj(rng, now, home, soul, case=None, earlier=None, env=None, tools=None):
    d = case or local_cases(rng, now)
    tools = d.get("tools") if tools is None else tools
    own_env = env is None
    if own_env:
        env, st = new_env_with(now, tools=tools)
    else:
        env.tools = list(tools or [])
        st = None
    for a in d["setup"]:
        env.act(a)
    t = Traj(header(now, home, soul, tools=tools, earlier=earlier), d["user"])
    t.model("Plan", d["plan"])
    if "ask" in d:
        t.model("Ask", d["ask"][0])
        t.user(d["ask"][1])
    if d["act"] is None:                      # nothing to run: the tool is not there
        t.model("Deliver", d["deliver"](""))
        truth = {"family": "local", "kind": d["kind"], "user": d["user"], "act": None, "setup": d["setup"],
                 "ask": d.get("ask"), "tools": tools}
        if own_env:
            drop_state(st)
        return t, truth
    r = t.act(env, d["act"])
    bad = r.lower().startswith(("how ", "i could", "unknown", "there is no", "no timer", "nothing", "no ", "sorry")) or "failed" in r.lower() or "?" in r
    if bad and d["kind"] not in ("timer_cancel", "timer_left", "agenda", "list_read", "reminders", "notes"):
        t.model("Judge", "The tool did not do it: " + r.rstrip(".") + ". Tell the user instead of pretending.")
        t.model("Deliver", "I could not do that: " + r.rstrip(".") + ".")
    else:
        t.model("Deliver", d["deliver"](r))
    truth = {"family": "local", "kind": d["kind"], "user": d["user"], "act": d["act"], "setup": d["setup"],
             "ask": d.get("ask"), "tools": tools}
    if own_env:
        drop_state(st)
    return t, truth


# ---------------------------------------------------------------- follow ups
def followup_traj(rng, now, home, soul, facts_by_subject):
    """Two turns in one round: the second refers to the first ("make it 20
    minutes instead", "and friday?", "who composed the music for it"), or is
    unrelated, and the model says which. The second turn is its own
    document with the first turn compressed into one Earlier line."""
    tools = rand_tools(rng)
    env, st = new_env_with(now, tools=tools)
    # a third of the second turns are unrelated to the first: the memory lines
    # must not turn every short request into a follow up
    kind = rng.choice(["timer", "alarm", "list", "agenda", "weather", "lights", "fact", "fact",
                       "unrelated", "unrelated", "unrelated", "unrelated"])
    base = None
    if kind == "timer":
        d1 = rng.choice(pools.durations("en")); d2 = rng.choice(pools.durations("en"))
        base = (f"set a timer for {d1}", f'timer("{d1}")', lambda r: r)
        fu = rng.choice([(f"make it {d2} instead", "Follow up to the timer: cancel it and set the new length.", [f"cancel_timer()", f'timer("{d2}")'], lambda rs: f"Timer changed to {d2}."),
                         ("cancel that", "Follow up: cancel the timer I just set.", ["cancel_timer()"], lambda rs: "Timer cancelled."),
                         ("how long is left", "Follow up: the remaining time on that timer.", ["timer_left()"], lambda rs: rs[-1])])
    elif kind == "alarm":
        t1, t2 = rng.choice(pools.times("en")), rng.choice(pools.times("en")); day = rng.choice(["tomorrow", "monday", "friday"])
        base = (f"wake me up at {t1} {day}", f'alarm("{t1}", "{day}")', lambda r: r)
        fu = rng.choice([(f"make it {t2}", "Follow up to the alarm: replace it with the new time.", ["cancel_alarm()", f'alarm("{t2}", "{day}")'], lambda rs: rs[-1]),
                         ("cancel it", "Follow up: cancel that alarm.", ["cancel_alarm()"], lambda rs: rs[-1])])
    elif kind == "list":
        ln = rng.choice(["shopping", "grocery", "packing"]); items = rng.sample(P("ITEM"), 2); more = rng.choice([x for x in P("ITEM") if x not in items])
        base = (f"add {items[0]} and {items[1]} to the {ln} list", f'list_add("{ln}", {json.dumps(items)})', lambda r: f"Added {items[0]}, {items[1]} to the {ln} list.")
        fu = rng.choice([(f"and {more}", f"Follow up: add that to the same list, the {ln} list.", [f'list_add("{ln}", ["{more}"])'], lambda rs: f"Added {more} to the {ln} list."),
                         (f"also {more} please", f"Follow up: add that to the same list, the {ln} list.", [f'list_add("{ln}", ["{more}"])'], lambda rs: f"Added {more} to the {ln} list."),
                         ("whats on it now", f"Follow up: read the {ln} list back.", [f'list_read("{ln}")'], lambda rs: f"Your {ln} list: " + rs[-1].split(":", 1)[-1].strip() + ".")])
    elif kind == "agenda":
        d1, d2 = rng.sample(["today", "tomorrow", "monday", "friday", "wednesday"], 2)
        for _ in range(rng.randrange(0, 3)):
            env.act(f'event("{rng.choice(P("EVENT"))}", "{d1}", "{rng.choice(pools.times("en"))}")')
        base = (f"what do i have {d1}", f'agenda("{d1}")', lambda r: (f"Nothing in the calendar for {d1}, you are free." if "Nothing" in r else f"On {d1}: " + "; ".join(x.strip() for x in r.splitlines()[1:]) + "."))
        fu = ((f"and {d2}?", f"Follow up: the same question for {d2}.", [f'agenda("{d2}")'],
               lambda rs: (f"Nothing in the calendar for {d2}, you are free." if "Nothing" in rs[-1] else f"On {d2}: " + "; ".join(x.strip() for x in rs[-1].splitlines()[1:]) + ".")))
    elif kind == "weather":
        place, place2 = rng.sample(HOMES, 2); d1, d2 = rng.sample(["today", "tomorrow", "the day after tomorrow", "friday"], 2)
        base = (f"whats the weather {d1} in {place}", f'weather("{place}", "{d1}")', lambda r: weather_summary(r, place, d1))
        fu = rng.choice([(f"and {d2}?", f"Follow up: the same place, {d2}.", [f'weather("{place}", "{d2}")'], lambda rs: weather_summary(rs[-1], place, d2)),
                         (f"and in {place2}?", f"Follow up: the same day, {place2}.", [f'weather("{place2}", "{d1}")'], lambda rs: weather_summary(rs[-1], place2, d1))])
    elif kind == "lights":
        room, room2 = rng.sample(P("ROOM"), 2); lv, lv2 = rng.sample(["full", "high", "half", "soft", "low"], 2)
        base = (f"{room} lights {lv}", f'lights("{lv}", "{room}")', lambda r: r + ".")
        fu = rng.choice([(f"make it {lv2}", f"Follow up: the same room, {lv2}.", [f'lights("{lv2}", "{room}")'], lambda rs: rs[-1] + "."),
                         (f"and the {room2}", f"Follow up: the same level in the {room2}.", [f'lights("{lv}", "{room2}")'], lambda rs: rs[-1] + "."),
                         ("turn them off", f"Follow up: the {room} lights off.", [f'lights("off", "{room}")'], lambda rs: rs[-1] + ".")])
    elif kind == "fact":
        pairs = [("director", "composer", "who composed the music for it", "film music composer", "the composer"),
                 ("composer", "director", "and who directed it", "film director", "the director"),
                 ("capital", "currency", "and what currency do they use", "currency", "the currency"),
                 ("currency", "capital", "and its capital?", "capital", "the capital")]
        r1, r2, q2, key2, noun2 = rng.choice(pairs)
        cands = [s for s, rels in facts_by_subject.items() if r1 in rels and r2 in rels]
        if not cands:
            drop_state(st)
            return None, None
        subj = rng.choice(cands)
        f1, f2 = facts_by_subject[subj][r1], facts_by_subject[subj][r2]
        t1, truth1 = fact_traj(f1, rng, now, home, soul, "direct")
        if t1 is None or truth1.get("answer") is None:
            drop_state(st)
            return None, None
        said1 = next(l[len("Deliver: "):] for l, m in t1.lines if l.startswith("Deliver:"))
        act1 = next(l[len("Act: "):] for l, m in reversed(t1.lines) if l.startswith("Act:"))
        earlier = [(truth1["user"], act1, said1)]
        t = Traj(header(now, home, soul, tools=tools, earlier=earlier), q2)
        t.model("Plan", f"Follow up about {subj} from the earlier turn: search {noun2} of it.")
        r = t.act(env, f'search("{subj} {key2}")')
        items = env.last_results
        a2 = f2["answer"]; cands2 = [a2] + [x for x in f2.get("aliases", []) if x]
        hit = next((i for i, x in enumerate(items) if any(found_in(c, x["snippet"]) for c in cands2)), None) if items and "error" not in items[0] else None
        if hit is None:
            page_i = next((i for i, x in enumerate(items) if norm(subj) in norm(x["title"])), None) if items and "error" not in items[0] else None
            if page_i is not None:
                t.model("Judge", f"The results mention {subj} but none states {noun2}. Open result {page_i + 1}, the page itself.")
                page = t.act(env, f"open({page_i + 1})")
                a_found = next((c for c in cands2 if found_in(c, page)), None)
                if a_found:
                    t.model("Judge", f'The page says: "{fragment(a_found, page, rel=r2)}", so {noun2} is {a_found}. That answers it.')
                    t.model("Deliver", SAY[r2].format(s=subj, a=a_found) + f" (Wikipedia: {items[page_i]['title']}).")
                    drop_state(st)
                    return t, {"family": "followup", "kind": "fact", "user": q2, "earlier": earlier, "answer": a2, "aliases": f2.get("aliases", []), "subject": subj}
            t.model("Judge", f"Not stated in the results; say so rather than guess.")
            t.model("Deliver", f"I could not find {noun2} of {subj} in the search results.")
            drop_state(st)
            return t, {"family": "followup", "kind": "fact", "user": q2, "earlier": earlier, "answer": None, "subject": subj}
        a_found = next(c for c in cands2 if found_in(c, items[hit]["snippet"]))
        t.model("Judge", f'Result {hit + 1} states it: "{fragment(a_found, items[hit]["snippet"], rel=r2)}", so {noun2} is {a_found}. That answers it.')
        t.model("Deliver", SAY[r2].format(s=subj, a=a_found) + f" (Wikipedia: {items[hit]['title']}).")
        drop_state(st)
        return t, {"family": "followup", "kind": "fact", "user": q2, "earlier": earlier, "answer": a2, "aliases": f2.get("aliases", []), "subject": subj}
    else:   # unrelated second turn after any base
        c1 = local_cases(rng, now)
        while c1["kind"] in ("tool",) or "ask" in c1 or c1["act"] is None:
            c1 = local_cases(rng, now)
        for a in c1["setup"]:
            env.act(a)
        base = (c1["user"], c1["act"], c1["deliver"])
        c2 = local_cases(rng, now)
        while c2["kind"] in ("tool", c1["kind"]) or "ask" in c2 or c2["act"] is None or c2["setup"]:
            c2 = local_cases(rng, now)
        fu = (c2["user"], "New request, not related to the earlier turn: " + c2["plan"][0].lower() + c2["plan"][1:], [c2["act"]], lambda rs, c2=c2: c2["deliver"](rs[-1]))
    # turn one for real, so the environment carries its state into turn two
    u1, act1, dl1 = base
    r1 = env.act(act1)
    said1 = dl1(r1)
    earlier = [(u1, act1, said1)]
    u2, plan2, acts2, dl2 = fu
    t = Traj(header(now, home, soul, tools=tools, earlier=earlier), u2)
    t.model("Plan", plan2)
    rs = []
    for k, a in enumerate(acts2):
        rs.append(t.act(env, a))
        if k + 1 < len(acts2):
            t.model("Judge", "Done; now the second step.")
    t.model("Deliver", dl2(rs))
    drop_state(st)
    return t, {"family": "followup", "kind": kind, "user": u2, "earlier": earlier, "setup": [act1], "acts": acts2, "tools": tools}


def compound_traj(rng, now, home, soul):
    a, b = local_cases(rng, now), local_cases(rng, now)
    if "ask" in a or "ask" in b or a["kind"] == b["kind"] or a["setup"] or b["setup"] or a["act"] is None or b["act"] is None or "tool" in (a["kind"], b["kind"]):
        return None, None
    env, st = new_env(now)
    user = f"{a['user']} {rng.choice(['and', 'and then', 'then', 'and also', ', also'])} {b['user']}"
    t = Traj(header(now, home, soul), user)
    t.model("Plan", "Two requests: do the first, then the second, then report both.")
    r1 = t.act(env, a["act"])
    t.model("Judge", "First done. Now the second.")
    r2 = t.act(env, b["act"])
    d1, d2 = a["deliver"](r1).rstrip("."), b["deliver"](r2).rstrip(".")
    t.model("Deliver", f"{d1}. {d2}.")
    drop_state(st)
    return t, {"family": "compound", "user": user, "acts": [a["act"], b["act"]]}


def umbrella_traj(rng, now, home, soul, vehicle, places):
    """An outdoor appointment one to six hours away: check rain at the hours before."""
    place = home if rng.random() < 0.7 else rng.choice(places)["city"]
    day = rng.choice(["today", "today", "tomorrow"])
    hour = rng.randrange(now.hour + 3, 22) if day == "today" and now.hour < 19 else rng.randrange(8, 21)
    if day == "today" and now.hour >= 19:
        day = "tomorrow"
    ev = rng.choice(["meeting", "dentist appointment", "lunch", "coffee with Maria", "client visit", "gym session", "football"])
    where = f" in {place}" if place != home else ""
    user = rng.choice([f"i have a {ev} at {hour}:00 {day}{where}, anything i should know",
                       f"{ev} at {hour}:00 {day}{where}, do i need to take anything",
                       f"im going out for a {ev} {day} at {hour}:00{where}, check the weather for me",
                       f"before my {ev} {day} {hour}:00{where}, should i take the car"])
    t = Traj(header(now, home, soul), user)
    env, st = new_env(now)
    t.model("Plan", f"Going out at {hour}:00 {day}: check the forecast for {place}, especially one and two hours before.")
    r = t.act(env, f'weather("{place}", "{day}")')
    if r.startswith("no forecast"):
        t.model("Judge", "No forecast came back, so I cannot judge the rain; say so.")
        t.model("Deliver", f"I could not get a forecast for {place}, so I cannot tell you about rain before your {ev}.")
        drop_state(st)
        return t, {"family": "umbrella", "user": user, "expected": "no_forecast"}
    rh = rain_hours(place, day, now)
    if rh is None:
        drop_state(st)
        return None, None
    rainy, allh = rh
    window = [f"{h:02d}:00" for h in (hour - 2, hour - 1, hour) if 0 <= h <= 23]
    wet = [(h, allh.get(h, 0)) for h in window if h in rainy]
    if wet:
        h, p = wet[0]
        advice = {"scooter": "You ride a scooter, so take the car or an umbrella.", "car": "Take an umbrella from the car to the door.",
                  "bike": "You cycle, so take a rain jacket or the bus.", "metro": "Take an umbrella for the walk from the metro.",
                  "walk": "Take an umbrella."}[vehicle]
        t.model("Judge", f"Rain probability is {p}% at {h}, within two hours of the {ev}. That matters for someone who {soul}.")
        t.model("Deliver", f"Rain is likely around {h} in {place} ({p}%), right before your {ev} at {hour}:00. {advice}")
        expected = "rain"
    else:
        temps = [allh.get(h, 0) for h in window]
        t.model("Judge", f"Rain probability stays under 40% at {', '.join(window)}, so no rain gear is needed.")
        t.model("Deliver", f"No rain expected before your {ev} at {hour}:00 {day} in {place}. Go as you are.")
        expected = "dry"
    drop_state(st)
    return t, {"family": "umbrella", "user": user, "place": place, "day": day, "hour": hour, "expected": expected, "vehicle": vehicle}


OTHER = [
    ("tell me a joke", "Not a task or a fact I can check; say what I do instead.",
     "I do not have jokes. I can set timers and alarms, keep lists and notes, check facts, play music and read your calendar."),
    ("write me a poem about the sea", "Not something I do; say so briefly.",
     "I do not write poems. I can look things up, set reminders, play music and manage your calendar."),
    ("how are you", "Small talk; answer briefly and offer help.", "Fine and ready. What do you need?"),
    ("thanks", "Acknowledge.", "You are welcome."),
    ("what am i thinking right now", "Unknowable; say so.", "I cannot know that. Ask me for something I can check or do."),
    ("what will the stock market do next year", "Unknowable, and searching would only find opinions; say so.",
     "Nobody can tell you that reliably, and I will not guess. I can look up what it did in the past if you want."),
    ("who is the smartest person alive", "There is no fact to find, only opinions; say so.",
     "There is no measured answer to that, only opinions. I can look up specific people or records if you name one."),
    ("hello", "Greeting; answer and offer help.", "Hello. What can I do for you?"),
    ("good morning", "Greeting; answer briefly.", "Good morning. Anything you need?"),
    ("can you see me", "No camera and no vision; say so.", "No, I cannot see anything. I work with text and the tools on this device."),
]


def other_traj(rng, now, home, soul):
    u, plan, ans = rng.choice(OTHER)
    t = Traj(header(now, home, soul), u)
    t.model("Plan", plan)
    t.model("Deliver", ans)
    return t, {"family": "other", "user": u, "expected": ans}


# ---------------------------------------------------------------- driver
def build(args):
    rng = random.Random(args.seed)
    AGENT.mkdir(parents=True, exist_ok=True)
    facts = [json.loads(l) for l in (AGENT / "facts.jsonl").open(encoding="utf-8")]
    hist = re.compile(r"dynasty|kingdom|empire|governorate|occupation|republic of|league|sultanate|bishopric|"
                      r"state of|duchy|khanate|emirate|principality|province of|colony|mandate|protectorate|"
                      r"confederation|caliphate|viceroyalty|commonwealth of|period|era\b|ancient|neo-|civili|"
                      r"city-state|polity|tribe|people|culture|pattin|assyria|babylon|hittite|\b\d{3,4}\b", re.I)
    facts = [f for f in facts if f["rel"] not in ("capital", "currency", "capital_region") or not hist.search(f["subject"])]
    songs = [json.loads(l) for l in (AGENT / "songs.jsonl").open(encoding="utf-8")]
    frozen = AGENT / "test_frozen.jsonl"
    if frozen.exists():
        fz = [json.loads(l) for l in frozen.open(encoding="utf-8")]
        fz_facts = {(x["rel"], norm(x["subject"])) for x in fz if x["family"] == "fact"}
        fz_songs = {norm(x["title"]) for x in fz if x["family"] == "song"}
        facts = [f for f in facts if (f["rel"], norm(f["subject"])) not in fz_facts]
        songs = [s for s in songs if norm(s["title"]) not in fz_songs]
        print(f"frozen benchmark: {len(fz_facts)} facts and {len(fz_songs)} songs kept out of training", flush=True)
    facts_by_subject = {}
    for f in facts:
        facts_by_subject.setdefault(f["subject"], {})[f["rel"]] = f
    places = [json.loads(l) for l in (AGENT / "places.jsonl").open(encoding="utf-8")]
    rng.shuffle(facts)
    rng.shuffle(songs)
    # facts whose good query is already cached go first: a rerun then spends
    # its Wikipedia budget only on new facts
    from .agent_tools import cache
    sc = cache("search")
    cached = [f for f in facts if sc.get(f"5|{f['subject']} {KEY[f['rel']]}".lower()) is not None]
    fresh = [f for f in facts if sc.get(f"5|{f['subject']} {KEY[f['rel']]}".lower()) is None]
    facts = cached + fresh
    print(f"facts: {len(cached)} with cached searches, {len(fresh)} new", flush=True)
    library_pool = [f"{s['title']} - {s['artist']}" for s in songs]
    base_now = datetime.now().replace(second=0, microsecond=0)

    def session(r):
        now = base_now + timedelta(hours=r.randrange(0, 30), minutes=r.choice([0, 15, 30, 45]))
        soul, vehicle = r.choice(SOUL)
        return now, r.choice(HOMES), soul, vehicle

    out = []       # (text, spans, truth, split)
    lock = threading.Lock()
    stats = {}
    from .agent_tools import cache as _cache
    tcache = _cache(f"traj_{GEN_VERSION}")     # finished fact trajectories, so a rerun only builds what changed

    def keep(t, truth, split):
        if t is None:
            return
        text = t.text()
        if len(text) > args.max_chars:
            stats["too_long"] = stats.get("too_long", 0) + 1
            return
        with lock:
            out.append((text, t.spans(), truth, split))
            k = truth["family"]
            stats[k] = stats.get(k, 0) + 1

    # facts, in parallel (network bound)
    n_facts = min(args.facts, len(facts))
    n_test = min(args.test_facts, n_facts // 10)

    def do_fact(job):
        # every fact is asked args.phrasings times, each with its own wording,
        # session and shape; all copies share the fact's split, so a held out
        # fact never leaks into training through another phrasing
        i, ph = job
        f = facts[i]
        r = random.Random(args.seed * 7919 + i * 31 + ph)
        now, home, soul, _ = session(r)
        u = r.random()
        shape = "direct" if u < 0.5 else ("bad_first" if u < 0.85 else "open")
        split = "test" if i < n_test else ("val" if i < n_test + n_facts // 50 else "train")
        ckey = f"{f['rel']}|{f['subject']}|{ph}|{shape}|{args.seed}"
        hit = tcache.get(ckey)
        if hit is not None:
            text, spans, truth = hit
            if truth.get("answer") is None and r.random() > args.giveup_keep:
                stats["giveup_dropped"] = stats.get("giveup_dropped", 0) + 1
                return
            if len(text) > args.max_chars:
                stats["too_long"] = stats.get("too_long", 0) + 1
                return
            with lock:
                out.append((text, [tuple(x) for x in spans], truth, split))
                stats["fact"] = stats.get("fact", 0) + 1
                stats["cached"] = stats.get("cached", 0) + 1
            return
        try:
            t, truth = fact_traj(f, r, now, home, soul, shape)
        except Exception as e:      # noqa: BLE001
            stats["fact_error"] = stats.get("fact_error", 0) + 1
            return
        if t is not None:
            tcache.put(ckey, [t.text(), t.spans(), truth])
        if truth and truth.get("answer") is None and r.random() > args.giveup_keep:
            # facts our own procedure cannot find are common with obscure
            # subjects; keep only enough of them for the honest give up lesson
            stats["giveup_dropped"] = stats.get("giveup_dropped", 0) + 1
            return
        keep(t, truth, split)

    t0 = time.time()
    jobs = [(i, ph) for i in range(n_facts) for ph in range(args.phrasings)]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for k, _ in enumerate(ex.map(do_fact, jobs)):
            if k % 8000 == 0:
                print(f"facts {k}/{len(jobs)}  {time.time() - t0:.0f}s  {stats}", flush=True)

    # songs, sequential and throttled (YouTube)
    n_songs = min(args.songs, len(songs))
    n_test_s = min(args.test_songs, n_songs // 10)
    for i in range(n_songs):
        r = random.Random(args.seed * 104729 + i)
        now, home, soul, _ = session(r)
        lib = r.sample(library_pool, min(25, len(library_pool)))
        try:
            t, truth = song_traj(songs[i], r, now, home, soul, lib, r.random() < 0.4)
        except Exception as e:      # noqa: BLE001
            stats["song_error"] = stats.get("song_error", 0) + 1
            continue
        keep(t, truth, "test" if i < n_test_s else ("val" if i < n_test_s + n_songs // 50 else "train"))
        if i == 0:
            print(f"songs {n_songs}  {time.time() - t0:.0f}s", flush=True)
        time.sleep(args.yt_delay)

    # local, compound, agenda and umbrella, other
    for i in range(args.local):
        r = random.Random(args.seed * 15485863 + i)
        now, home, soul, _ = session(r)
        t, truth = local_traj(r, now, home, soul)
        keep(t, truth, "test" if i < args.local // 20 else ("val" if i < args.local // 10 else "train"))
    for i in range(args.compound):
        r = random.Random(args.seed * 32452843 + i)
        now, home, soul, _ = session(r)
        t, truth = compound_traj(r, now, home, soul)
        keep(t, truth, "test" if i < args.compound // 20 else ("val" if i < args.compound // 10 else "train"))
    for i in range(args.umbrella):
        r = random.Random(args.seed * 49979687 + i)
        now, home, soul, vehicle = session(r)
        now = base_now.replace(hour=r.randrange(7, 16), minute=0)
        try:
            t, truth = umbrella_traj(r, now, home, soul, vehicle, places[:300])
        except Exception as e:      # noqa: BLE001
            stats["umbrella_error"] = stats.get("umbrella_error", 0) + 1
            continue
        keep(t, truth, "test" if i < args.umbrella // 20 else ("val" if i < args.umbrella // 10 else "train"))
        if i == 0:
            print(f"umbrella {args.umbrella}  {time.time() - t0:.0f}s", flush=True)
    for i in range(args.followup):
        r = random.Random(args.seed * 86028121 + i)
        now, home, soul, _ = session(r)
        try:
            t, truth = followup_traj(r, now, home, soul, facts_by_subject)
        except Exception as e:      # noqa: BLE001
            stats["followup_error"] = stats.get("followup_error", 0) + 1
            continue
        keep(t, truth, "test" if i < args.followup // 20 else ("val" if i < args.followup // 10 else "train"))
        if i == 0:
            print(f"followup {args.followup}  {time.time() - t0:.0f}s", flush=True)
    for i in range(args.other):
        r = random.Random(args.seed * 67867967 + i)
        now, home, soul, _ = session(r)
        t, truth = other_traj(r, now, home, soul)
        keep(t, truth, "test" if i < args.other // 20 else "train")

    print("built", len(out), stats, f"{time.time() - t0:.0f}s", flush=True)
    write(out, args)


def write(out, args):
    from tokenizers import Tokenizer
    OUTDIR = Path(args.out) if args.out else AGENT
    OUTDIR.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.from_file(str(ROOT / "pretrain" / "tokenizer.json"))
    eot = tok.token_to_id(EOT)
    rng = random.Random(args.seed + 1)
    rng.shuffle(out)
    for split in ("train", "val"):
        ids, mask, texts = [], [], []
        for text, spans, truth, sp in out:
            if sp != split:
                continue
            enc = tok.encode(text)
            m = np.zeros(len(enc.ids), dtype=np.uint8)
            for k, (a, b) in enumerate(enc.offsets):
                if any(a < e and b > s for s, e in spans):
                    m[k] = 1
            ids.extend(enc.ids)
            mask.extend(m.tolist())
            ids.append(eot)
            mask.append(1)          # the model must learn to stop after Deliver
            texts.append(text)
        np.save(OUTDIR / f"{split}.npy", np.array(ids, dtype=np.uint16))
        np.save(OUTDIR / f"{split}_mask.npy", np.array(mask, dtype=np.uint8))
        (OUTDIR / f"{split}.txt").write_text((EOT + "\n").join(texts), encoding="utf-8", newline="\n")
        print(f"{split}: {len(texts)} trajectories, {len(ids)/1e6:.2f}M tokens, {100*sum(mask)/max(1,len(mask)):.0f}% model tokens")
    with (OUTDIR / "test.jsonl").open("w", encoding="utf-8") as f:
        n = 0
        for text, spans, truth, sp in out:
            if sp == "test":
                f.write(json.dumps(truth, ensure_ascii=False) + "\n")
                n += 1
    print(f"test: {n} held out tasks")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--facts", type=int, default=40000)
    ap.add_argument("--test_facts", type=int, default=1000)
    ap.add_argument("--songs", type=int, default=4000)
    ap.add_argument("--test_songs", type=int, default=300)
    ap.add_argument("--local", type=int, default=16000)
    ap.add_argument("--compound", type=int, default=4000)
    ap.add_argument("--umbrella", type=int, default=2000)
    ap.add_argument("--other", type=int, default=1200)
    ap.add_argument("--followup", type=int, default=3000)
    ap.add_argument("--out", default=None, help="output directory (default corpus/agent); one per round lets generation overlap training")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--yt_delay", type=float, default=0.4)
    ap.add_argument("--max_chars", type=int, default=3600)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--giveup_keep", type=float, default=0.15, help="share of unfindable facts kept as honest give ups")
    ap.add_argument("--phrasings", type=int, default=1, help="trajectories per fact, each worded and shaped differently")
    build(ap.parse_args())


if __name__ == "__main__":
    main()
