"""Facts with known answers for the search and read trajectories.

    python -m pretrain.facts          -> corpus/agent/facts.jsonl, songs.jsonl, places.jsonl

Source: DBpedia's SPARQL endpoint (structured Wikipedia, free, no key,
answers in seconds), with Wikidata's query service only for the two "who is
currently" relations, one request a minute, because that service was rate
limited to that during an outage when this was written. Each fact is
{rel, subject, answer, aliases}; the answer is the label the model has to
find inside real Wikipedia search results. Notability proxy on DBpedia: the
number of language labels the subject has (an article in 25 languages is one
people ask about).
"""
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

from .agent_tools import _CTX, UA

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "corpus" / "agent"
DBP = "https://dbpedia.org/sparql"
WDQS = "https://query.wikidata.org/sparql"

EN = "FILTER(lang(?sl)='en' && lang(?al)='en')"

# rel: (pattern binding ?s and ?a (a resource) or ?av (a literal), minimum number of
# language labels the subject has (the fame proxy: an article in 25 languages is
# a film people search for), total wanted)
DBPEDIA = {
    "capital": ("?s a dbo:Country; dbo:capital ?a.", 5, 600),
    "currency": ("?s a dbo:Country; dbo:currency ?a.", 5, 600),
    "capital_region": ("?s a dbo:AdministrativeRegion; dbo:capital ?a.", 15, 4000),
    "country_of_city": ("?s a dbo:City; dbo:country ?a.", 35, 5000),
    "director": ("?s a dbo:Film; dbo:director ?a.", 25, 7000),
    "composer": ("?s a dbo:Film; dbo:musicComposer ?a.", 25, 3000),
    "release_year": ("?s a dbo:Film; dbo:releaseDate ?av.", 30, 5000),
    "author": ("?s a dbo:Book; dbo:author ?a.", 20, 4000),
    "height": ("?s a dbo:Mountain; dbo:elevation ?av.", 15, 3000),
    "founded": ("?s a dbo:Company; dbo:foundingYear ?av.", 25, 3000),
    "developer": ("?s a dbo:Software; dbo:developer ?a.", 20, 3000),
    "born_year": ("?s a dbo:Person; dbo:birthDate ?av.", 40, 9000),
    "died_year": ("?s a dbo:Person; dbo:deathDate ?av.", 40, 5000),
    "birthplace": ("?s a dbo:Person; dbo:birthPlace ?a. ?a a dbo:City.", 40, 5000),
}
# the sport is the class of the athlete; the answer gets aliases for the judge
SPORTS = {"dbo:TennisPlayer": ("tennis", ["tennis"]), "dbo:BasketballPlayer": ("basketball", ["basketball"]),
          "dbo:SoccerPlayer": ("football", ["football", "footballer", "soccer"]),
          "dbo:Swimmer": ("swimming", ["swimming", "swimmer"]), "dbo:Boxer": ("boxing", ["boxing", "boxer"]),
          "dbo:Cyclist": ("cycling", ["cycling", "cyclist"]), "dbo:GolfPlayer": ("golf", ["golf", "golfer"]),
          "dbo:Cricketer": ("cricket", ["cricket", "cricketer"]), "dbo:BaseballPlayer": ("baseball", ["baseball"]),
          "dbo:IceHockeyPlayer": ("ice hockey", ["ice hockey", "hockey"]),
          "dbo:FormulaOneRacer": ("Formula One", ["formula one", "formula 1", "racing driver"])}

WIKIDATA = {
    "head_gov": """SELECT ?sl ?al WHERE { ?s wdt:P31 wd:Q6256; wikibase:sitelinks ?n. FILTER(?n > 100)
        ?s p:P6 ?st. ?st ps:P6 ?a. FILTER NOT EXISTS { ?st pq:P582 ?end }
        ?s rdfs:label ?sl. ?a rdfs:label ?al. FILTER(lang(?sl)='en' && lang(?al)='en') }""",
    "head_state": """SELECT ?sl ?al WHERE { ?s wdt:P31 wd:Q6256; wikibase:sitelinks ?n. FILTER(?n > 100)
        ?s p:P35 ?st. ?st ps:P35 ?a. FILTER NOT EXISTS { ?st pq:P582 ?end }
        ?s rdfs:label ?sl. ?a rdfs:label ?al. FILTER(lang(?sl)='en' && lang(?al)='en') }""",
}


def log(*a):
    print(*a, flush=True)


def query(endpoint, q, timeout=120):
    url = endpoint + "?" + urllib.parse.urlencode({"query": q, "format": "json"})
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/sparql-results+json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout, context=_CTX).read())["results"]["bindings"]


def dbp_rows(pattern, min_labels, want, page=2000):
    """Rows of (subject label, answer) for subjects with at least min_labels
    language labels; GROUP BY does the counting on the server."""
    literal = "?av" in pattern
    sel = "?sl ?av" if literal else "?sl ?al"
    labels = "?s rdfs:label ?sl. FILTER(lang(?sl)='en')" if literal else f"?s rdfs:label ?sl. ?a rdfs:label ?al. {EN}"
    out, off = [], 0
    while off < want:
        q = (f"SELECT {sel} (COUNT(?l) AS ?n) WHERE {{ {pattern} ?s rdfs:label ?l. {labels} }} "
             f"GROUP BY {sel} HAVING (COUNT(?l) >= {min_labels}) LIMIT {page} OFFSET {off}")
        try:
            got = query(DBP, q)
        except Exception as e:      # noqa: BLE001
            log("    failed:", str(e)[:100])
            time.sleep(5)
            break
        out += got
        if len(got) < page:
            break
        off += page
    return out, literal


def clean_label(s):
    s = re.sub(r"\s*\([^)]*\)\s*$", "", s).strip()      # "Inception (film)" -> "Inception"
    return s


def ok(s):
    return s and 2 <= len(s) <= 70 and not re.search(r"[<>{}|\\^~\[\]`]", s) and not re.match(r"^[\d\W]+$", s)


def year_of(v):
    m = re.match(r"^(-?\d{1,4})", v)
    if not m:
        return None
    y = int(m.group(1))
    return str(y) if 1000 <= y <= 2026 else None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    f = (OUT / "facts.jsonl").open("w", encoding="utf-8")
    seen, total = set(), 0

    def emit(rel, subj, ans, aliases=None):
        nonlocal total
        subj, ans = clean_label(subj), ans.strip()
        if not (ok(subj) and ok(ans)) or subj.lower() == ans.lower() or (rel, subj) in seen:
            return 0
        seen.add((rel, subj))
        f.write(json.dumps({"rel": rel, "subject": subj, "answer": ans, "aliases": aliases or []}, ensure_ascii=False) + "\n")
        total += 1
        return 1

    for rel, (pattern, min_labels, want) in DBPEDIA.items():
        t0 = time.time()
        got, literal = dbp_rows(pattern, min_labels, want)
        n = 0
        for r in got:
            subj = r["sl"]["value"]
            if literal:
                v = r["av"]["value"]
                if rel in ("born_year", "died_year", "release_year", "founded"):
                    v = year_of(v)
                elif rel == "height":
                    try:
                        v = str(int(round(float(v))))
                    except ValueError:
                        v = None
                if not v:
                    continue
                n += emit(rel, subj, v)
            else:
                n += emit(rel, subj, clean_label(r["al"]["value"]))
        f.flush()
        log(f"{rel:16s} {n:5d} facts  {time.time() - t0:5.0f}s  total {total}")

    t0, n = time.time(), 0
    for cls, (sport, aliases) in SPORTS.items():
        got, _ = dbp_rows(f"?s a {cls}. BIND('x' AS ?av)", 30, 500, page=500)
        for r in got:
            n += emit("sport", r["sl"]["value"], sport, aliases)
    f.flush()
    log(f"{'sport':16s} {n:5d} facts  {time.time() - t0:5.0f}s  total {total}")

    for rel, q in WIKIDATA.items():
        t0, n = time.time(), 0
        try:
            got = query(WDQS, q)
            for r in got:
                n += emit(rel, r["sl"]["value"], r["al"]["value"])
        except Exception as e:      # noqa: BLE001
            log("    wikidata failed:", str(e)[:100])
        f.flush()
        log(f"{rel:16s} {n:5d} facts  {time.time() - t0:5.0f}s  total {total}")
        time.sleep(65)
    f.close()

    # songs: title and performer
    got, _ = dbp_rows("?s a dbo:Single; dbo:artist ?a.", 12, 6000)
    songs = {}
    for r in got:
        t, a = clean_label(r["sl"]["value"]), clean_label(r["al"]["value"])
        if ok(t) and ok(a) and (t, a) not in songs:
            songs[(t, a)] = {"title": t, "artist": a}
    with (OUT / "songs.jsonl").open("w", encoding="utf-8") as g:
        for x in songs.values():
            g.write(json.dumps(x, ensure_ascii=False) + "\n")
    log("songs:", len(songs))

    got, _ = dbp_rows("?s a dbo:City; dbo:country ?a; dbo:populationTotal ?pop. FILTER(?pop > 400000)", 40, 2000)
    places = {}
    for r in got:
        c = clean_label(r["sl"]["value"])
        if ok(c) and c not in places:
            places[c] = {"city": c, "country": clean_label(r["al"]["value"])}
    with (OUT / "places.jsonl").open("w", encoding="utf-8") as g:
        for x in places.values():
            g.write(json.dumps(x, ensure_ascii=False) + "\n")
    log("places:", len(places))


if __name__ == "__main__":
    main()
