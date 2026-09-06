"""The reasoning layer runtime: tools, agent loop and a llama.cpp server.

It does not know facts. It knows how to check them: given a question, it
decides which tool to call (including web_search), reads the result, and
answers. It runs through llama.cpp on the CPU with four threads, which is how
a phone would run it. The weights are whatever GGUF BSLM_GGUF points to; the
default is our own model exported by pretrain/export_gguf.py. A pretrained
0.6B model was used once as a yardstick and is not part of the plan.
"""
import html
import json
import os
import re
import subprocess
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

from . import skills

ROOT = Path(__file__).resolve().parent.parent
SERVER = ROOT / "tools" / "llama" / "llama-server.exe"
MODEL = Path(os.environ.get("BSLM_GGUF", ROOT / "models" / "56m-fineweb-q8.gguf"))
PORT = int(os.environ.get("BSLM_LLM_PORT", "8089"))
THREADS = int(os.environ.get("BSLM_LLM_THREADS", "4"))

SYSTEM = (
    "You are Bee, an assistant running on a phone. Today is {date}. "
    "You do not know current facts. When the user asks about people, events, "
    "prices, weather, news, or anything you cannot be sure of, call web_search "
    "and answer only from the results, in one or two sentences, naming the source. "
    "Use the other tools to act on the device; a request may need several tools. "
    "Reply in English. Never invent facts. Be brief."
)


def tool(name, desc, **props):
    req = [k for k, v in props.items() if not v.get("optional")]
    for v in props.values():
        v.pop("optional", None)
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props, "required": req}}}


S = lambda d, **kw: {"type": "string", "description": d, **kw}

TOOLS = [
    tool("web_search", "Search the web. Use for any fact you do not know.",
         query=S("search query")),
    tool("set_timer", "Start a countdown timer.", duration=S("e.g. '5 minutes'")),
    tool("cancel_timer", "Cancel the running timer."),
    tool("timer_left", "How much time is left on the timer."),
    tool("set_alarm", "Set an alarm.", time=S("e.g. '7:30' or '7 am'"),
         date=S("e.g. 'tomorrow'", optional=True)),
    tool("create_reminder", "Remind the user to do something.",
         task=S("what to do"), date=S("when, e.g. 'Friday'", optional=True),
         time=S("e.g. '9:00'", optional=True)),
    tool("list_reminders", "Read the user's reminders."),
    tool("add_to_list", "Add items to a list.",
         items={"type": "array", "items": {"type": "string"}, "description": "items"},
         list_name=S("e.g. 'shopping list'", optional=True)),
    tool("read_list", "Read a list.", list_name=S("e.g. 'shopping list'", optional=True)),
    tool("create_note", "Save a note.", text=S("the note")),
    tool("read_notes", "Read the saved notes."),
    tool("add_calendar_event", "Add an event to the calendar.",
         title=S("event"), date=S("date", optional=True), time=S("time", optional=True)),
    tool("query_calendar", "What is on the calendar.", date=S("date", optional=True)),
    tool("calculate", "Evaluate arithmetic.", expression=S("e.g. '348 / 12'")),
    tool("convert_units", "Convert units or currency.",
         amount=S("number"), from_unit=S("e.g. 'km'"), to_unit=S("e.g. 'miles'")),
    tool("current_time", "The current time and date."),
    tool("lights", "Control lights.", state=S("'on' or 'off'"),
         room=S("room", optional=True), color=S("color", optional=True)),
    tool("device", "Switch a device on or off.", device=S("e.g. 'tv'"), state=S("'on' or 'off'")),
    tool("set_volume", "Set the volume 0 to 100.", level=S("number")),
    tool("play_music", "Play music.", query=S("song, artist, genre or playlist")),
]


def web_search(query, n=5):
    url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote_plus(query)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    try:
        page = urllib.request.urlopen(req, timeout=15).read().decode("utf-8", "replace")
    except Exception as e:
        return f"search failed: {e}"
    titles = re.findall(r'class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>', page, re.S)
    snips = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', page, re.S)
    out = []
    for i, ((href, title), snip) in enumerate(zip(titles, snips)):
        if i >= n:
            break
        m = re.search(r"uddg=([^&]+)", href)
        link = urllib.parse.unquote(m.group(1)) if m else href
        clean = lambda s: html.unescape(re.sub(r"<[^>]+>", "", s)).strip()
        out.append(f"{i + 1}. {clean(title)}: {clean(snip)} ({link})")
    return "\n".join(out) or "no results"


class Reasoner:
    def __init__(self, speak=print, autostart=True):
        self.bot = skills.Assistant(speak=speak)
        self.proc = None
        self.last_timings = {}
        if autostart:
            self.start()

    # ---------------- server ----------------
    def alive(self):
        try:
            return urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2).status == 200
        except Exception:
            return False

    def start(self, timeout=90):
        if self.alive():
            return
        if not SERVER.exists() or not MODEL.exists():
            raise FileNotFoundError("llama-server.exe or the GGUF model is missing")
        flags = 0x08000000 if os.name == "nt" else 0          # CREATE_NO_WINDOW
        self.proc = subprocess.Popen(
            [str(SERVER), "-m", str(MODEL), "--port", str(PORT), "--host", "127.0.0.1",
             "--jinja", "-c", "4096", "-t", str(THREADS), "-ngl", "0",
             "--cache-reuse", "256", "--log-disable"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=flags)
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.alive():
                return
            time.sleep(0.5)
        raise RuntimeError("llama-server did not come up")

    def stop(self):
        if self.proc:
            self.proc.terminate()
            self.proc = None

    def chat(self, messages):
        body = json.dumps({
            "messages": messages, "tools": TOOLS, "temperature": 0.1,
            "max_tokens": 300, "chat_template_kwargs": {"enable_thinking": False},
        }).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{PORT}/v1/chat/completions", data=body,
            headers={"Content-Type": "application/json"})
        r = json.loads(urllib.request.urlopen(req, timeout=180).read())
        self.last_timings = r.get("timings", {})
        return r["choices"][0]["message"]

    # ---------------- tools ----------------
    def execute(self, name, a):
        b = self.bot
        g = lambda k, d="": str(a.get(k) or d)
        try:
            if name == "web_search":
                return web_search(g("query"))
            if name == "set_timer":
                return b.timer_set({"duration": g("duration"), "_text": g("duration")})
            if name == "cancel_timer":
                return b.timer_cancel({})
            if name == "timer_left":
                return b.timer_query({})
            if name == "set_alarm":
                return b.alarm_set({"time": g("time"), "date": g("date", "tomorrow")})
            if name == "create_reminder":
                return b.reminder_create({"task": g("task"), "date": g("date"), "time": g("time")})
            if name == "list_reminders":
                return b.reminder_list({})
            if name == "add_to_list":
                items = a.get("items") or []
                return b.list_add({"item": [str(i) for i in items], "list_name": g("list_name", "shopping list")})
            if name == "read_list":
                return b.list_read({"list_name": a.get("list_name")})
            if name == "create_note":
                return b.note_create({"note": g("text")})
            if name == "read_notes":
                return b.note_read({})
            if name == "add_calendar_event":
                return b.calendar_create({"title": g("title"), "date": g("date"), "time": g("time")})
            if name == "query_calendar":
                return b.calendar_query({"date": a.get("date")})
            if name == "calculate":
                return b.math_calculate({"expression": g("expression")})
            if name == "convert_units":
                return b.unit_convert({"amount": g("amount"), "unit_from": g("from_unit"), "unit_to": g("to_unit")})
            if name == "current_time":
                return b.time_query({"_text": ""})
            if name == "lights":
                st = g("state", "on").lower()
                return b.light_control({"room": g("room", "all"), "color": a.get("color"),
                                        "_text": f"turn {st} the lights"})
            if name == "device":
                st = g("state", "on").lower()
                return b.device_control({"device": g("device"), "_text": f"turn {st} the {g('device')}"})
            if name == "set_volume":
                return b.volume_set({"level": g("level"), "_text": g("level")})
            if name == "play_music":
                return b.music_play({"song": g("query")})
            return f"unknown tool {name}"
        except Exception as e:
            return f"tool error: {e}"

    # ---------------- agent loop ----------------
    def run(self, text, history=None, max_rounds=4):
        """Returns (answer, trace). trace lists every tool call made."""
        messages = [{"role": "system", "content": SYSTEM.format(date=datetime.now().strftime("%A %d %B %Y"))}]
        messages += history or []
        messages.append({"role": "user", "content": text})
        trace = []
        t0 = time.time()
        for _ in range(max_rounds):
            msg = self.chat(messages)
            calls = msg.get("tool_calls") or []
            if not calls:
                answer = (msg.get("content") or "").strip()
                return answer or "I could not work that out.", {"tools": trace, "seconds": round(time.time() - t0, 1)}
            messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
            for c in calls:
                fn = c["function"]
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = self.execute(fn["name"], args)
                trace.append((fn["name"], args, result[:120]))
                messages.append({"role": "tool", "tool_call_id": c.get("id", fn["name"]),
                                 "name": fn["name"], "content": result})
        return "I did what I could, but I ran out of steps.", {"tools": trace, "seconds": round(time.time() - t0, 1)}
