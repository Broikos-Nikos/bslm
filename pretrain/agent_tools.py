"""The model's hands: real tools shared by the trajectory generator and the runtime.

Every result the model ever sees in training comes from these functions, run
for real (Wikipedia search and page extracts, YouTube result pages, Open-Meteo
forecasts, the local skills with a fresh state file), so the runtime shows it
nothing it has not seen the shape of. Network answers are cached on disk in
corpus/agent/cache/ so a generation run can be repeated offline.

Actions, as the model writes them on an `Act:` line:

    search("q")            web style result list (Wikipedia search, 5 hits)
    open(n)                text of result n of the last search
    weather("place", "day")   hourly forecast, day = today, tomorrow, weekday or date
    youtube("q")           top videos for a query
    library("q")           the user's own music library
    play(n)                play item n of the last youtube or library list
    timer("10 minutes")  cancel_timer()  timer_left()
    alarm("7:30", "tomorrow")  cancel_alarm()
    remind("task", "friday", "9:00")  reminders()
    list_add("shopping", ["milk", "eggs"])  list_read("shopping")
    note("text")  notes()
    event("dentist", "monday", "10:00")  agenda("tomorrow")
    calc("348 / 12")  convert(5, "km", "miles")  time()
    lights("off", "kitchen")  lights("half", "living room")  lights("movie", "living room")
    switch("water heater", "on")   a smart plug or appliance, on or off
    tool("qr file receiver", "open")   one of the tools listed in the header
    device("tv", "off")  volume(30)
    ask("which list?")     hands the turn back to the user
"""
import ast
import html
import json
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "corpus" / "agent" / "cache"
UA = "bslm-research/0.1 (small language model experiment; github.com/Broikos-Nikos/bslm)"
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


# ---------------------------------------------------------------- cache
class Cache:
    """One jsonl file per tool, loaded once, appended on every new answer."""

    def __init__(self, name):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.path = CACHE_DIR / f"{name}.jsonl"
        self.d = {}
        self.lock = threading.Lock()
        if self.path.exists():
            for line in self.path.open(encoding="utf-8"):
                try:
                    k, v = json.loads(line)
                    self.d[k] = v
                except ValueError:
                    pass

    def get(self, k):
        return self.d.get(k)

    def put(self, k, v):
        with self.lock:
            self.d[k] = v
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps([k, v], ensure_ascii=False) + "\n")
        return v


_caches = {}


def cache(name):
    if name not in _caches:
        _caches[name] = Cache(name)
    return _caches[name]


def ssl_context():
    """The Windows certificate store here rejects the Wikimedia chain as expired;
    certifi's bundle is current, so every fetch uses it."""
    import ssl
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


_CTX = ssl_context()


_throttle = {"lock": threading.Lock(), "last": 0.0, "api_blocked_until": 0.0}
WIKI_INTERVAL = float(__import__("os").environ.get("BSLM_WIKI_INTERVAL", "0.8"))   # seconds between Wikipedia requests, all threads


def _polite(url):
    """Wikimedia asks for a few requests per second at most, serially; one
    global gap between requests to their hosts keeps every thread in line."""
    if "wikipedia.org" not in url:
        return
    with _throttle["lock"]:
        wait = _throttle["last"] + WIKI_INTERVAL - time.time()
        if wait > 0:
            time.sleep(wait)
        _throttle["last"] = time.time()


class RateLimited(RuntimeError):
    pass


import itertools as _itertools
_PROXY = __import__("os").environ.get("BSLM_PI_PROXY", "").strip()
# both openers carry the certifi context; the Windows store rejects the
# Wikimedia chain, and that must hold whether the request goes direct or
# through the Pi proxy (HTTPS over the proxy is a CONNECT tunnel)
_direct_opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=_CTX))
_proxy_opener = (urllib.request.build_opener(urllib.request.HTTPSHandler(context=_CTX),
                                             urllib.request.ProxyHandler({"http": _PROXY, "https": _PROXY}))
                 if _PROXY else None)
_opener_turn = _itertools.count()


def _opener(url):
    # split only Wikipedia load; DBpedia, YouTube and Open-Meteo stay direct
    if _proxy_opener is not None and "wikipedia.org" in url and next(_opener_turn) % 2:
        return _proxy_opener
    return _direct_opener


def _get(url, headers=None, timeout=20, tries=3):
    h = {"User-Agent": UA, "Accept-Language": "en"}
    h.update(headers or {})
    err = None
    for i in range(tries):
        try:
            op = _opener(url)
            if op is _direct_opener:      # the proxy has its own budget; only pace direct Wikipedia calls
                _polite(url)
            req = urllib.request.Request(url, headers=h)
            return op.open(req, timeout=timeout).read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            if e.code == 429:
                # back off hard, the limit is per client and shared by every thread
                if i + 1 < tries:
                    time.sleep(4 * 2 ** i)
                    continue
                raise RateLimited(f"429 from {url.split('/')[2]}")
            err = e
            time.sleep(1.5 * (i + 1))
        except Exception as e:      # noqa: BLE001
            err = e
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"fetch failed: {err}")


def api_ok():
    return time.time() >= _throttle["api_blocked_until"]


def api_blocked():
    """After a 429 the API stays off for half an hour; the HTML pages carry on."""
    _throttle["api_blocked_until"] = time.time() + 1800


def _clean(s):
    return html.unescape(re.sub(r"<[^>]+>", "", s)).replace("\n", " ").strip()


# ---------------------------------------------------------------- search and open
def lookup_search(query, n=5):
    """DBpedia Lookup: a keyword search over Wikipedia articles, same result
    shape, no rate limit worth mentioning, weaker ranking. Cached."""
    key = f"lookup|{n}|{query.strip().lower()}"
    c = cache("search")
    hit = c.get(key)
    if hit is not None:
        return hit
    url = "https://lookup.dbpedia.org/api/search?" + urllib.parse.urlencode({"query": query, "maxResults": n, "format": "JSON"})
    try:
        docs = json.loads(_get(url, headers={"Accept": "application/json"})).get("docs", [])
        items = []
        for d in docs[:n]:
            res = (d.get("resource") or [""])[0]
            title = _clean((d.get("label") or [res.rsplit("/", 1)[-1].replace("_", " ")])[0])
            items.append({"title": title, "snippet": _clean((d.get("comment") or [""])[0])[:260],
                          "url": "https://en.wikipedia.org/wiki/" + res.rsplit("/", 1)[-1]})
    except Exception as e:      # noqa: BLE001
        return [{"error": str(e)}]
    return c.put(key, items)


def wiki_search(query, n=5, backend=None):
    """Result list in web search shape: title, snippet, url. Cached."""
    if backend == "lookup":
        return lookup_search(query, n)
    key = f"{n}|{query.strip().lower()}"
    c = cache("search")
    hit = c.get(key)
    if hit is not None:
        return hit
    items = None
    if api_ok():
        url = ("https://en.wikipedia.org/w/api.php?action=query&list=search&format=json"
               f"&srlimit={n}&srsearch=" + urllib.parse.quote_plus(query))
        try:
            data = json.loads(_get(url))
            items = [{"title": r["title"], "snippet": _clean(r.get("snippet", "")),
                      "url": "https://en.wikipedia.org/wiki/" + urllib.parse.quote(r["title"].replace(" ", "_"))}
                     for r in data.get("query", {}).get("search", [])]
        except RateLimited:
            api_blocked()
        except Exception as e:      # noqa: BLE001
            return [{"error": str(e)}]
    if items is None:
        # the same search through the site's own results page, which is not
        # under the API's rate limit rule
        url = "https://en.wikipedia.org/w/index.php?" + urllib.parse.urlencode(
            {"search": query, "fulltext": 1, "ns0": 1, "limit": n})
        try:
            page = _get(url, headers={"Accept": "text/html"})
            heads = re.findall(r'<div class="mw-search-result-heading"><a href="/wiki/([^"]+)"[^>]*>(.*?)</a>', page, re.S)
            snips = re.findall(r'<div class="searchresult">(.*?)</div>', page, re.S)
            items = [{"title": _clean(t), "snippet": _clean(sn), "url": "https://en.wikipedia.org/wiki/" + h}
                     for (h, t), sn in zip(heads, snips)][:n]
        except Exception as e:      # noqa: BLE001
            return [{"error": str(e)}]
    return c.put(key, items)


def infobox_rows(page, limit=14):
    """'Key: value' lines from the article's infobox, short rows only."""
    m = re.search(r'<table[^>]*class="[^"]*infobox[^"]*"[^>]*>(.*?)</table>', page, re.S)
    if not m:
        return []
    rows = []
    for k, v in re.findall(r"<th[^>]*>(.*?)</th>\s*<td[^>]*>(.*?)</td>", m.group(1), re.S):
        k, v = _clean(re.sub(r"<br\s*/?>", ", ", k)), _clean(re.sub(r"<(?:br|/li|/p)[^>]*>", ", ", v))
        v = re.sub(r"\[\d+\]|\s*,\s*,", "", v).strip(" ,")
        if 2 <= len(k) <= 32 and 1 <= len(v) <= 120 and not re.search(r"^\W|website|image|caption|coordinates|native name", k, re.I):
            rows.append(f"{k}: {v}")
        if len(rows) >= limit:
            break
    return rows


def wiki_extract(title, chars=900):
    """The article as the model reads it: infobox rows, then the lead
    paragraphs, from the mobile page. Cached."""
    key = title.strip().lower()
    c = cache("page3")      # page3: one infobox row per line, a shorter lead
    hit = c.get(key)
    if hit is not None:
        return hit
    old = cache("page2").get(key)
    if old is not None and not old.startswith("could not open"):
        # the same page as fetched before, rows were joined with "; " and the
        # lead followed the last row after ". "; split it back, no refetch
        parts = old.split("; ")
        rows, lead = [], old
        if len(parts) > 1 and ": " in parts[0]:
            last, sep, tail = parts[-1].partition(". ")
            rows, lead = parts[:-1] + [last], tail if sep else ""
        elif ": " in parts[0] and ". " in parts[0]:
            first, _, tail = parts[0].partition(". ")
            rows, lead = [first], tail
        return c.put(key, ("\n".join(rows) + "\n" if rows else "") + lead[:chars])
    text = None
    if False:
        url = ("https://en.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=1&exintro=1"
               f"&exchars={chars}&format=json&redirects=1&titles=" + urllib.parse.quote(title))
        try:
            pages = json.loads(_get(url))["query"]["pages"]
            text = next(iter(pages.values())).get("extract", "") or ""
            text = re.sub(r"\s+", " ", text).strip()
        except RateLimited:
            api_blocked()
        except Exception as e:      # noqa: BLE001
            text = f"could not open the page: {e}"
    if text is None:
        # the mobile article page: infobox rows, then the lead paragraphs
        url = "https://en.m.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"))
        try:
            page = _get(url, headers={"Accept": "text/html"})
            rows = infobox_rows(page)
            m = re.search(r'<div class="mw-parser-output">(.*)', page, re.S)
            body = (m.group(1) if m else page).split("<h2", 1)[0]
            body = re.sub(r"(?s)<table.*?</table>", " ", body)
            paras = [_clean(p) for p in re.findall(r"<p[^>]*>(.*?)</p>", body, re.S)]
            lead = re.sub(r"\[\d+\]", "", " ".join(p for p in paras if p))
            lead = re.sub(r"\s+", " ", lead).strip()[:chars]
            # one row per line: the model reads "Music by: Hans Zimmer" as its own
            # line instead of a name lost in a long sentence
            text = ("\n".join(rows) + "\n" if rows else "") + lead
        except Exception as e:      # noqa: BLE001
            text = f"could not open the page: {e}"
    if text.startswith("could not open"):
        return text            # not cached, so a later run can retry
    return c.put(key, text)


def open_url(url, chars=1200):
    m = re.match(r"https?://en\.wikipedia\.org/wiki/(.+)$", url)
    if m:
        return wiki_extract(urllib.parse.unquote(m.group(1)).replace("_", " "), chars)
    c = cache("page")
    hit = c.get(url)
    if hit is not None:
        return hit
    try:
        page = _get(url)
        page = re.sub(r"(?is)<(script|style|nav|header|footer).*?</\1>", " ", page)
        text = re.sub(r"\s+", " ", _clean(page))[:chars]
    except Exception as e:      # noqa: BLE001
        text = f"could not open the page: {e}"
    return c.put(url, text)


def fmt_results(items):
    if not items:
        return "no results"
    if items and "error" in items[0]:
        return "search failed: " + items[0]["error"]
    return "\n".join(f"{i + 1}. {r['title']}: {r['snippet'][:220]} ({r['url']})" for i, r in enumerate(items))


# ---------------------------------------------------------------- weather
def geocode(place):
    key = place.strip().lower()
    c = cache("geocode")
    hit = c.get(key)
    if hit is not None:
        return hit
    url = ("https://geocoding-api.open-meteo.com/v1/search?count=1&language=en&format=json&name="
           + urllib.parse.quote_plus(place))
    try:
        res = json.loads(_get(url)).get("results") or []
        if not res:
            return c.put(key, None)
        r = res[0]
        return c.put(key, {"name": r["name"], "country": r.get("country", ""),
                           "lat": r["latitude"], "lon": r["longitude"], "tz": r.get("timezone", "auto")})
    except Exception:      # noqa: BLE001
        return None


def resolve_day(day, now):
    """'today', 'tomorrow', a weekday name or an ISO date -> date, else None."""
    d = (day or "today").strip().lower()
    if d in ("today", "now", ""):
        return now.date()
    if d == "tomorrow":
        return now.date() + timedelta(days=1)
    if d in ("day after tomorrow", "the day after tomorrow"):
        return now.date() + timedelta(days=2)
    if d in WEEKDAYS:
        delta = (WEEKDAYS.index(d) - now.weekday()) % 7
        return now.date() + timedelta(days=delta)
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})$", d)
    if m:
        return date(int(m[1]), int(m[2]), int(m[3]))
    return None


def forecast(place, day, now):
    """Hourly temperature and rain probability for one day, as one text block.

    Open-Meteo gives 7 days from today; the cache key is place and date so a
    generation run and a later benchmark see the same numbers."""
    loc = geocode(place)
    if not loc:
        return f"no forecast: unknown place '{place}'"
    d = resolve_day(day, now)
    if d is None:
        return f"no forecast: I do not understand the day '{day}'"
    if not (0 <= (d - now.date()).days <= 6):
        return f"no forecast: {d.isoformat()} is outside the 7 day forecast"
    key = f"{loc['lat']:.2f},{loc['lon']:.2f}|{d.isoformat()}"
    c = cache("forecast")
    hit = c.get(key)
    if hit is None:
        url = ("https://api.open-meteo.com/v1/forecast?hourly=temperature_2m,precipitation_probability,weather_code"
               f"&timezone=auto&start_date={d.isoformat()}&end_date={d.isoformat()}"
               f"&latitude={loc['lat']}&longitude={loc['lon']}")
        try:
            h = json.loads(_get(url))["hourly"]
            hit = c.put(key, [[t[11:16], temp, p, w] for t, temp, p, w in
                              zip(h["time"], h["temperature_2m"], h["precipitation_probability"], h["weather_code"])])
        except Exception as e:      # noqa: BLE001
            return f"no forecast: {e}"
    rows = [r for r in hit if r[0] >= "06:00"]
    line = ", ".join(f"{t} {round(temp)}C {p}%" for t, temp, p, _ in rows)
    rain = [r for r in hit if (r[2] or 0) >= 40]
    summary = "rain likely " + ", ".join(r[0] for r in rain[:6]) if rain else "no rain expected"
    return f"{loc['name']}, {loc['country']}, {d.isoformat()}: {line}. {summary}."


def rain_hours(place, day, now, threshold=40):
    """Hours with rain probability at or above the threshold, from the same cache."""
    loc = geocode(place)
    d = resolve_day(day, now)
    if not loc or d is None:
        return None
    hit = cache("forecast").get(f"{loc['lat']:.2f},{loc['lon']:.2f}|{d.isoformat()}")
    if hit is None:
        forecast(place, day, now)
        hit = cache("forecast").get(f"{loc['lat']:.2f},{loc['lon']:.2f}|{d.isoformat()}")
    if hit is None:
        return None
    return {r[0]: r[2] for r in hit if (r[2] or 0) >= threshold}, {r[0]: r[2] for r in hit}


# ---------------------------------------------------------------- youtube
def youtube_search(query, n=5):
    key = f"{n}|{query.strip().lower()}"
    c = cache("youtube")
    hit = c.get(key)
    if hit is not None:
        return hit
    url = "https://www.youtube.com/results?search_query=" + urllib.parse.quote_plus(query)
    try:
        page = _get(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/128.0 Safari/537.36",
                                  "Cookie": "CONSENT=YES+cb; SOCS=CAI"})
        m = re.search(r"var ytInitialData = (\{.*?\});</script>", page, re.S)
        items = []
        if m:
            def walk(o):
                if isinstance(o, dict):
                    if "videoRenderer" in o:
                        v = o["videoRenderer"]
                        try:
                            items.append({"id": v["videoId"],
                                          "title": "".join(r["text"] for r in v["title"]["runs"]),
                                          "channel": v.get("ownerText", {}).get("runs", [{}])[0].get("text", ""),
                                          "length": v.get("lengthText", {}).get("simpleText", "")})
                        except (KeyError, IndexError):
                            pass
                    for x in o.values():
                        walk(x)
                elif isinstance(o, list):
                    for x in o:
                        walk(x)
            walk(json.loads(m.group(1)))
        items = items[:n]
    except Exception as e:      # noqa: BLE001
        return [{"error": str(e)}]
    if not items:
        return items          # not cached: probably a block page, retry another time
    return c.put(key, items)


def fmt_videos(items):
    if not items:
        return "no results"
    if "error" in items[0]:
        return "search failed: " + items[0]["error"]
    return "\n".join(f"{i + 1}. {v['title']} ({v['channel']}, {v['length'] or 'live'})" for i, v in enumerate(items))


# ---------------------------------------------------------------- the environment
def parse_act(line):
    """'search("x")' -> ("search", ["x"]); None when the line is not an action."""
    m = re.match(r"\s*([a-z_]+)\s*\((.*)\)\s*$", line.strip(), re.S)
    if not m:
        return None
    name, inner = m.group(1), m.group(2).strip()
    if not inner:
        return name, []
    try:
        args = ast.literal_eval("(" + inner + ",)")
    except (ValueError, SyntaxError):
        return name, [inner.strip("\"'")]
    return name, list(args)


class Env:
    """Runs one action at a time and keeps the session state.

    state_file: where the local skills persist (a temp file in the generator,
    data/state.json in the app). now: the clock the session lives in.
    library: the user's music library as 'Title - Artist' strings."""

    def __init__(self, state_file, now=None, library=None, online=True, tools=None):
        self.tools = list(tools or [])      # (name, description) registered by the wrapper
        from bslm import skills
        self.skills = skills
        skills.STATE = Path(state_file)
        self.now = now or datetime.now()
        if now is not None:
            # the skills stamp reminders, agenda answers and time() with
            # datetime.now(); in a generated session the clock is the header's
            fixed = now

            class FrozenClock(datetime):
                @classmethod
                def now(cls, tz=None):
                    return cls.fromtimestamp(fixed.timestamp(), tz) if tz else cls.fromtimestamp(fixed.timestamp())
            skills.datetime = FrozenClock
        self.bot = skills.Assistant(speak=lambda *a, **k: None)
        self.library = library or []
        self.online = online
        self.last_results = []
        self.last_videos = []
        self.last_library = []
        self.last_list = None       # "youtube" or "library"
        self.playing = None
        self.next_backend = None    # the generator routes one search elsewhere
        self.home = "Athens"        # used when a weather question names no place

    # helpers
    def _nth(self, items, n, what):
        try:
            i = int(n)
        except (TypeError, ValueError):
            return None, f"{what}: give a result number"
        if not items:
            return None, f"{what}: there is no list to pick from"
        if not 1 <= i <= len(items):
            return None, f"{what}: there is no result {i}, only 1 to {len(items)}"
        return items[i - 1], None

    def act(self, line):
        p = parse_act(line)
        if not p:
            return "unknown action, use one of: search, open, weather, youtube, library, play, timer, alarm, remind, list_add, list_read, note, event, agenda, calc, convert, time, lights, device, volume, ask"
        name, a = p
        g = lambda i, d="": str(a[i]) if len(a) > i and a[i] is not None else d
        try:
            return self._run(name, a, g)
        except Exception as e:      # noqa: BLE001
            return f"{name} failed: {e}"

    LEVELS = {"full": 100, "max": 100, "maximum": 100, "bright": 100, "high": 80, "on": 100,
              "half": 50, "medium": 50, "mid": 50, "low": 30, "soft": 30, "dim": 30, "warm": 30,
              "night": 10, "minimum": 10, "min": 10, "off": 0}
    SCENES = ("movie", "cinema", "reading", "cozy", "relax", "party", "dinner", "focus", "work", "sleep")

    def lights(self, level, room, color=None):
        """Lights by level (full, high, half, soft, a percentage), by scene
        (movie, reading, cozy: the preset the home app carries under that
        name), or simply on and off."""
        lv = level.strip().lower().rstrip("%")
        if lv in self.SCENES or lv.endswith(" mode"):
            scene = lv.replace(" mode", "")
            home = self.bot.state["home"]
            home["lights"][room] = {"on": True, "scene": scene, "brightness": None, "color": None}
            self.skills.save_state(self.bot.state)
            return f"Lights in {room}: {scene} scene"
        if lv.isdigit():
            pct = max(0, min(100, int(lv)))
        elif lv in self.LEVELS:
            pct = self.LEVELS[lv]
        else:
            return f"lights: I do not know the level '{level}'; use full, high, half, soft, off, a percentage or a scene"
        r = self.bot.light_control({"room": room, "color": color, "brightness": f"{pct}%" if 0 < pct < 100 else None,
                                    "_text": "turn off the lights" if pct == 0 else "turn on the lights"})
        if 0 < pct < 100 and not lv.isdigit():
            r = r.replace("(brightness", f"({lv},")
        return r

    def _run(self, name, a, g):
        b = self.bot
        if name == "search":
            if not self.online:
                return "search failed: offline"
            self.last_results = wiki_search(g(0), backend=self.next_backend)
            self.next_backend = None
            return fmt_results(self.last_results)
        if name == "open":
            if self.last_results and "error" in self.last_results[0]:
                return "open: the last search failed, search again first"
            r, err = self._nth(self.last_results, g(0), "open")
            if err:
                return err
            return open_url(r["url"]) or "the page is empty"
        if name == "weather":
            if not self.online:
                return "no forecast: offline"
            place, day = g(0), g(1, "today")
            # trained on questions that always named a place, so a place-less
            # question ("whats the weather today") drops the day word into the
            # place slot; recover by using the home location
            if not place or resolve_day(place, self.now) is not None:
                if resolve_day(place, self.now) is not None and (not g(1) or g(1) == place):
                    day = place
                place = self.home
            return forecast(place, day, self.now)
        if name == "youtube":
            if not self.online:
                return "search failed: offline"
            self.last_videos = youtube_search(g(0))
            self.last_list = "youtube"
            return fmt_videos(self.last_videos)
        if name == "library":
            q = g(0).lower()
            words = [w for w in re.findall(r"\w+", q) if len(w) > 1]
            hits = [s for s in self.library if all(w in s.lower() for w in words)] if words else []
            self.last_library = hits[:5]
            self.last_list = "library"
            return "\n".join(f"{i + 1}. {s}" for i, s in enumerate(self.last_library)) or "no match in the library"
        if name == "play":
            items = self.last_videos if self.last_list == "youtube" else self.last_library
            r, err = self._nth(items, g(0), "play")
            if err:
                return err
            title = r["title"] if isinstance(r, dict) else r
            self.playing = title
            return f"Playing: {title}"
        if name == "timer":
            return b.timer_set({"duration": g(0), "_text": g(0)})
        if name == "cancel_timer":
            return b.timer_cancel({})
        if name == "timer_left":
            return b.timer_query({})
        if name == "alarm":
            return b.alarm_set({"time": g(0), "date": g(1, "tomorrow")})
        if name == "cancel_alarm":
            return b.alarm_cancel({})
        if name == "remind":
            return b.reminder_create({"task": g(0), "date": g(1), "time": g(2)})
        if name == "reminders":
            return b.reminder_list({})
        if name == "list_add":
            items = a[1] if len(a) > 1 and isinstance(a[1], (list, tuple)) else [g(1)]
            return b.list_add({"item": [str(i) for i in items if str(i)], "list_name": g(0, "shopping list")})
        if name == "list_read":
            return b.list_read({"list_name": g(0) or None})
        if name == "note":
            return b.note_create({"note": g(0)})
        if name == "notes":
            return b.note_read({})
        if name == "event":
            return b.calendar_create({"title": g(0), "date": g(1), "time": g(2)})
        if name == "agenda":
            return b.calendar_query({"date": g(0) or None})
        if name == "calc":
            return b.math_calculate({"expression": g(0)})
        if name == "convert":
            return b.unit_convert({"amount": g(0), "unit_from": g(1), "unit_to": g(2)})
        if name == "time":
            return b.time_query({"_text": ""})
        if name == "lights":
            return self.lights(g(0, "on").lower(), g(1, "all"), a[2] if len(a) > 2 else None)
        if name == "tool":
            want = g(0).strip().lower()
            for entry in self.tools:
                tname = entry[0]
                if want == tname.lower() or want in tname.lower() or tname.lower() in want:
                    # a registered tool may carry a command (data/tools.json "cmd");
                    # the wrapper runs it, the model only ever sees the text
                    cmd = entry[2] if len(entry) > 2 else None
                    if cmd:
                        try:
                            import subprocess
                            subprocess.Popen(cmd, shell=True)
                        except Exception as e:      # noqa: BLE001
                            return f"{tname}: could not run it ({e})"
                    return f"{tname}: {g(1, 'open')} done"
            names = ", ".join(t for t, _ in self.tools) or "none registered"
            return f"no tool named '{g(0)}'; the tools are: {names}"
        if name == "switch":
            st = g(1, "on").lower()
            on = st not in ("off", "close", "closed", "stop", "0")
            r = b.device_control({"device": g(0), "_text": f"turn {'on' if on else 'off'} the {g(0)}"})
            return r.replace("turned", "switched")
        if name == "device":
            st = g(1, "on").lower()
            return b.device_control({"device": g(0), "_text": f"turn {st} the {g(0)}"})
        if name == "volume":
            return b.volume_set({"level": g(0), "_text": g(0)})
        if name == "ask":
            return "(waiting for the user)"
        return f"unknown action {name}"
