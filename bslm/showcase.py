"""BSLM showcase: a dark, animated window that runs the shipped v1 model live
and shows what it is doing while it does it, the neurons pulsing as it thinks,
a tool lighting up for each action it takes, the smart lights changing, the
search and the answer streaming into the chat.

It is a thin wrapper: the brain is the real from-scratch 72M loop model driven
through bslm/agent.py, the same code the benchmark uses. Nothing here is faked.

Run it by double clicking run_showcase.bat, or:
    .venv\\Scripts\\python.exe -m bslm.showcase
"""
import os
import queue
import random
import sys
import threading
import time
import tkinter as tk
from pathlib import Path

from . import config

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------- palette
BG = "#0b0f1a"          # near black blue
PANEL = "#121829"
PANEL2 = "#0e1422"
EDGE = "#1e2740"
TEXT = "#e6ebff"
MUTE = "#7688b5"
ACCENT = "#3aa0ff"      # electric blue
ACCENT2 = "#8b5cff"     # violet
HOT = "#ff5d8f"         # pink
GOOD = "#37e0a6"        # mint
WARM = "#ffd27a"

TOOLS = [   # key, label, colour when firing
    ("search", "SEARCH", ACCENT), ("open", "READ", ACCENT), ("weather", "WEATHER", GOOD),
    ("youtube", "YOUTUBE", HOT), ("play", "PLAY", HOT), ("library", "LIBRARY", HOT),
    ("lights", "LIGHTS", WARM), ("switch", "PLUG", WARM), ("device", "DEVICE", WARM),
    ("timer", "TIMER", ACCENT2), ("alarm", "ALARM", ACCENT2), ("remind", "REMIND", ACCENT2),
    ("list_add", "LISTS", ACCENT2), ("note", "NOTES", ACCENT2), ("event", "CALENDAR", ACCENT2),
    ("agenda", "CALENDAR", ACCENT2), ("calc", "MATH", GOOD), ("convert", "UNITS", GOOD),
    ("tool", "TOOL", HOT), ("time", "CLOCK", MUTE), ("volume", "VOLUME", ACCENT2),
]
ALIAS = {"cancel_timer": "timer", "timer_left": "timer", "cancel_alarm": "alarm",
         "reminders": "remind", "list_read": "list_add", "notes": "note"}


class Neurons:
    """A little feed forward net drawn on a canvas. It breathes when idle and
    fires pulses left to right while the model is thinking."""

    def __init__(self, canvas):
        self.c = canvas
        self.layers = [6, 9, 9, 5]
        self.nodes = []          # (x, y, base_glow)
        self.node_layer = []
        self.edges = []          # (i, j, line_id)
        self.pulses = []         # (edge_index, t, colour)
        self.activity = 0.0
        self.phase = 0.0
        self._built = False

    def build(self, w, h):
        self.c.delete("all")
        self.cx, self.cy = w / 2, h / 2
        # a glowing orb behind the network, in the gradients.design style: an
        # off-centre highlight, a rim brighter than the middle, no rotation
        self.orbR = min(w, h) * 0.42
        hx, hy = self.cx - self.orbR * 0.18, self.cy - self.orbR * 0.22   # highlight offset
        steps = 30
        for i in range(steps, 0, -1):
            t = i / steps                       # 1 at the rim, 0 at the highlight
            # body: violet core warming to a bright cyan rim
            col = self._mix(self._mix(ACCENT2, "#1a2340", 0.35), ACCENT, max(0.0, (t - 0.55) / 0.45)) if t > 0.55 \
                else self._mix("#141c30", ACCENT2, 0.5 * (1 - t))
            r = self.orbR * t
            self.c.create_oval(hx - r, hy - r, hx + r, hy + r, outline="", fill=col)
        # the atmospheric rim, pulsed in step()
        self.rim = self.c.create_oval(self.cx - self.orbR, self.cy - self.orbR,
                                      self.cx + self.orbR, self.cy + self.orbR, outline=ACCENT, width=3)
        self.nodes, self.node_layer, self.edges = [], [], []
        cols = len(self.layers)
        margin_x, margin_y = 70, 40
        xs = [margin_x + (w - 2 * margin_x) * k / (cols - 1) for k in range(cols)]
        idx_by_layer = []
        for li, n in enumerate(self.layers):
            ids = []
            for k in range(n):
                y = margin_y + (h - 2 * margin_y) * (k + 0.5) / n
                self.nodes.append([xs[li], y, random.random()])
                self.node_layer.append(li)
                ids.append(len(self.nodes) - 1)
            idx_by_layer.append(ids)
        for li in range(cols - 1):
            for a in idx_by_layer[li]:
                for b in idx_by_layer[li + 1]:
                    if random.random() < 0.55:
                        x1, y1, _ = self.nodes[a]; x2, y2, _ = self.nodes[b]
                        lid = self.c.create_line(x1, y1, x2, y2, fill=EDGE, width=1)
                        self.edges.append([a, b, lid])
        self.node_items = []
        for (x, y, g) in self.nodes:
            self.node_items.append(self.c.create_oval(x - 6, y - 6, x + 6, y + 6, outline="", fill=PANEL))
        # a calm central core that breathes brighter while it thinks
        self.core = self.c.create_oval(self.cx - 10, self.cy - 10, self.cx + 10, self.cy + 10, outline="", fill=ACCENT)
        self._built = True

    def fire(self, strength=1.0):
        self.activity = min(1.4, self.activity + strength)
        for _ in range(int(6 * strength)):
            if self.edges:
                self.pulses.append([random.randrange(len(self.edges)), 0.0,
                                    random.choice([ACCENT, ACCENT2, GOOD, HOT])])

    def _mix(self, c1, c2, t):
        a = tuple(int(c1[i:i + 2], 16) for i in (1, 3, 5))
        b = tuple(int(c2[i:i + 2], 16) for i in (1, 3, 5))
        return "#%02x%02x%02x" % tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))

    def step(self):
        if not self._built:
            return
        self.phase += 0.045          # calm, no fast motion
        self.activity *= 0.94
        base = 0.25 + 0.75 * self.activity
        # breathing core
        import math as _m
        breath = 0.5 + 0.5 * _m.sin(self.phase * 1.3)
        cr = 9 + 5 * breath + 10 * self.activity
        self.c.coords(self.core, self.cx - cr, self.cy - cr, self.cx + cr, self.cy + cr)
        self.c.itemconfig(self.core, fill=self._mix(ACCENT, "#ffffff", 0.3 * self.activity))
        # the orb rim brightens and swells a touch while thinking
        if hasattr(self, "rim") and getattr(self, "orbR", 0):
            rr = self.orbR * (1.0 + 0.03 * breath + 0.06 * self.activity)
            self.c.coords(self.rim, self.cx - rr, self.cy - rr, self.cx + rr, self.cy + rr)
            self.c.itemconfig(self.rim, outline=self._mix(ACCENT, "#ffffff", 0.15 + 0.5 * self.activity),
                              width=2 + 2 * self.activity)
        for ni, (x, y, g) in enumerate(self.nodes):
            tw = 0.5 + 0.5 * (0.5 + 0.5 * __import__("math").sin(self.phase + g * 6))
            lvl = min(1.0, base * tw)
            col = self._mix(PANEL, ACCENT if self.node_layer[ni] % 2 == 0 else ACCENT2, lvl)
            r = 5 + 4 * lvl
            self.c.coords(self.node_items[ni], x - r, y - r, x + r, y + r)
            self.c.itemconfig(self.node_items[ni], fill=col)
        # edges dim glow with activity
        eg = self._mix(EDGE, ACCENT, 0.4 * self.activity)
        for e in self.edges:
            self.c.itemconfig(e[2], fill=eg)
        # pulses travel
        alive = []
        for p in self.pulses:
            p[1] += 0.05
            if p[1] < 1.0:
                a, b, _ = self.edges[p[0]]
                x1, y1, _ = self.nodes[a]; x2, y2, _ = self.nodes[b]
                x = x1 + (x2 - x1) * p[1]; y = y1 + (y2 - y1) * p[1]
                dot = self.c.create_oval(x - 3, y - 3, x + 3, y + 3, outline="", fill=p[2], tags="pulse")
                alive.append(p)
        self.c.delete("pulsedot")
        for p in alive:
            a, b, _ = self.edges[p[0]]
            x1, y1, _ = self.nodes[a]; x2, y2, _ = self.nodes[b]
            x = x1 + (x2 - x1) * p[1]; y = y1 + (y2 - y1) * p[1]
            self.c.create_oval(x - 3, y - 3, x + 3, y + 3, outline="", fill=p[2], tags="pulsedot")
        self.c.delete("pulse")
        self.pulses = alive


class Showcase(tk.Tk):
    def __init__(self, selftest=False):
        super().__init__()
        self.title("BSLM v1  .  a 72M model built from scratch")
        self.configure(bg=BG)
        self.geometry("1180x760")
        self.minsize(1000, 640)
        self.q = queue.Queue()
        self.cfg = config.load()
        self.name = self.cfg["name"]
        self.agent = None
        self.pending_ask = False
        self.busy = False
        self.tiles = {}
        self.lit = {}          # label -> (until_time, colour)
        self.lamps = []
        self._build()
        self.after(33, self._tick)
        if not selftest:
            threading.Thread(target=self._load_agent, daemon=True).start()

    # ---------------- layout
    def _panel(self, parent, **kw):
        return tk.Frame(parent, bg=PANEL, highlightbackground=EDGE, highlightthickness=1, **kw)

    def _build(self):
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        # left: specs
        left = self._panel(self); left.grid(row=0, column=0, sticky="ns", padx=(12, 6), pady=12)
        tk.Label(left, text="BSLM", bg=PANEL, fg=TEXT, font=("Segoe UI Semibold", 22)).pack(anchor="w", padx=16, pady=(16, 0))
        tk.Label(left, text="from scratch . no pretrained weights", bg=PANEL, fg=ACCENT, font=("Segoe UI", 9)).pack(anchor="w", padx=16)
        specs = [("model", "bslm-72m-v1"), ("params", "72.6 M"), ("file", "78 MB, Q8_0"),
                 ("arch", "Llama shaped decoder"), ("context", "4096 tokens"), ("runs on", "CPU, 4 threads"),
                 ("agent bench", "91% overall"), ("tasks", "timers, lists, notes,"),
                 ("", "calendar, maths, home,"), ("", "weather, music, search")]
        box = tk.Frame(left, bg=PANEL); box.pack(anchor="w", padx=16, pady=16, fill="x")
        for k, v in specs:
            row = tk.Frame(box, bg=PANEL); row.pack(anchor="w", fill="x", pady=2)
            tk.Label(row, text=k, bg=PANEL, fg=MUTE, font=("Consolas", 9), width=11, anchor="w").pack(side="left")
            tk.Label(row, text=v, bg=PANEL, fg=TEXT, font=("Consolas", 9), anchor="w").pack(side="left")
        # editable settings the assistant needs: place, the soul fact, the name
        tk.Label(left, text="SETTINGS  (editable)", bg=PANEL, fg=MUTE, font=("Consolas", 8)).pack(anchor="w", padx=16, pady=(10, 2))
        self.cfg_vars = {}
        for key, label in [("home", "place"), ("soul", "about you"), ("name", "name")]:
            row = tk.Frame(box.master, bg=PANEL); row.pack(anchor="w", fill="x", padx=16, pady=2)
            tk.Label(row, text=label, bg=PANEL, fg=MUTE, font=("Consolas", 9), width=9, anchor="w").pack(side="left")
            var = tk.StringVar(value=self.cfg[key])
            ent = tk.Entry(row, textvariable=var, bg=PANEL2, fg=TEXT, insertbackground=TEXT, bd=0,
                           highlightbackground=EDGE, highlightthickness=1, font=("Consolas", 9), width=16)
            ent.pack(side="left", ipady=3)
            self.cfg_vars[key] = var
        tk.Button(left, text="save settings", command=self._save_cfg, bg=PANEL2, fg=ACCENT,
                  activebackground=EDGE, activeforeground=ACCENT, bd=0, font=("Segoe UI", 9),
                  cursor="hand2").pack(anchor="w", padx=16, pady=(4, 0))

        self.status = tk.Label(left, text="waking the model...", bg=PANEL, fg=WARM, font=("Segoe UI", 9), wraplength=200, justify="left")
        self.status.pack(anchor="w", padx=16, pady=(8, 16))

        # center: neurons + chat + input
        center = tk.Frame(self, bg=BG); center.grid(row=0, column=1, sticky="nsew", padx=6, pady=12)
        center.columnconfigure(0, weight=1)
        center.rowconfigure(0, weight=1)   # neurons take all the room left over
        center.rowconfigure(1, weight=0)   # chat is a fixed height
        npanel = self._panel(center); npanel.grid(row=0, column=0, sticky="nsew")
        tk.Label(npanel, text="THINKING", bg=PANEL, fg=MUTE, font=("Consolas", 8)).pack(anchor="w", padx=10, pady=(6, 0))
        self.ncanvas = tk.Canvas(npanel, bg=PANEL, height=360, highlightthickness=0)
        self.ncanvas.pack(fill="both", expand=True, padx=6, pady=6)
        self.neurons = Neurons(self.ncanvas)
        self.ncanvas.bind("<Configure>", self._on_canvas)

        cpanel = self._panel(center, height=500); cpanel.grid(row=1, column=0, sticky="ew", pady=(8, 8))
        cpanel.pack_propagate(False)          # keep the chat exactly 500px tall
        scroll = tk.Scrollbar(cpanel, bd=0, highlightthickness=0, troughcolor=PANEL2)
        scroll.pack(side="right", fill="y")
        self.chat = tk.Text(cpanel, bg=PANEL, fg=TEXT, bd=0, highlightthickness=0, wrap="word",
                            font=("Segoe UI", 11), padx=14, pady=12, state="disabled", spacing1=2, spacing3=6,
                            yscrollcommand=scroll.set)
        self.chat.pack(side="left", fill="both", expand=True)
        scroll.config(command=self.chat.yview)
        self.chat.tag_configure("you", foreground=ACCENT, font=("Segoe UI Semibold", 11))
        self.chat.tag_configure("bee", foreground=GOOD, font=("Segoe UI Semibold", 11))
        self.chat.tag_configure("msg", foreground=TEXT)
        self.chat.tag_configure("dim", foreground=MUTE, font=("Consolas", 9))

        entry = tk.Frame(center, bg=BG); entry.grid(row=2, column=0, sticky="ew")
        entry.columnconfigure(0, weight=1)
        self.input = tk.Entry(entry, bg=PANEL2, fg=TEXT, insertbackground=TEXT, bd=0,
                              highlightbackground=EDGE, highlightthickness=1, font=("Segoe UI", 12))
        self.input.grid(row=0, column=0, sticky="ew", ipady=9, padx=(0, 8))
        self.input.bind("<Return>", lambda e: self._send())
        self.send_btn = tk.Button(entry, text="ask", command=self._send, bg=ACCENT, fg="#04101f",
                                  activebackground=ACCENT2, activeforeground="#04101f", bd=0,
                                  font=("Segoe UI Semibold", 11), width=8, cursor="hand2")
        self.send_btn.grid(row=0, column=1, ipady=6)

        # right: tools + lights + log
        right = self._panel(self); right.grid(row=0, column=2, sticky="ns", padx=(6, 12), pady=12)
        tk.Label(right, text="TOOLS", bg=PANEL, fg=MUTE, font=("Consolas", 8)).pack(anchor="w", padx=12, pady=(12, 4))
        grid = tk.Frame(right, bg=PANEL); grid.pack(padx=12)
        self.key_label = {key: label for key, label, col in TOOLS}
        label_col = {}
        for key, label, col in TOOLS:
            label_col.setdefault(label, col)
        for i, (label, col) in enumerate(label_col.items()):
            cell = tk.Label(grid, text=label, bg=PANEL2, fg=MUTE, font=("Consolas", 8, "bold"),
                            width=9, height=2, highlightbackground=EDGE, highlightthickness=1)
            cell.grid(row=i // 3, column=i % 3, padx=3, pady=3)
            self.tiles.setdefault(label, []).append((cell, col))

        tk.Label(right, text="SMART LIGHTS", bg=PANEL, fg=MUTE, font=("Consolas", 8)).pack(anchor="w", padx=12, pady=(16, 4))
        lampf = tk.Frame(right, bg=PANEL); lampf.pack(padx=12)
        self.lampcanvas = tk.Canvas(lampf, bg=PANEL, width=260, height=60, highlightthickness=0)
        self.lampcanvas.pack()
        for i in range(6):
            x = 26 + i * 42
            oid = self.lampcanvas.create_oval(x - 14, 16, x + 14, 44, outline=EDGE, width=2, fill=PANEL2)
            self.lamps.append(oid)
        self.lightlabel = tk.Label(right, text="off", bg=PANEL, fg=MUTE, font=("Consolas", 8))
        self.lightlabel.pack(anchor="w", padx=12)

        tk.Label(right, text="ACTION LOG", bg=PANEL, fg=MUTE, font=("Consolas", 8)).pack(anchor="w", padx=12, pady=(16, 4))
        self.log = tk.Text(right, bg=PANEL2, fg=MUTE, bd=0, highlightbackground=EDGE, highlightthickness=1,
                           width=34, height=16, wrap="word", font=("Consolas", 8), padx=8, pady=8, state="disabled")
        self.log.pack(padx=12, pady=(0, 12), fill="y", expand=True)
        self.log.tag_configure("act", foreground=ACCENT)
        self.log.tag_configure("res", foreground=MUTE)

    def _on_canvas(self, e):
        self.neurons.build(e.width, e.height)

    # ---------------- agent
    def _load_agent(self):
        try:
            from pretrain.agent_tools import Env
            from .agent import Agent, load_tools
            env = Env(ROOT / "data" / "showcase_state.json", tools=load_tools())
            a = Agent(env=env, home=self.cfg["home"], soul=self.cfg["soul"], tools=load_tools())
            # wrap the hands and the thinking so the UI sees each step live
            _act = a.env.act

            def act(line):
                self.q.put(("act_start", line))
                r = _act(line)
                self.q.put(("act_done", line, r))
                return r
            a.env.act = act
            _complete = a.complete

            def complete(prompt, max_tokens=160):
                self.q.put(("think",))
                return _complete(prompt, max_tokens)
            a.complete = complete
            self.agent = a
            self.q.put(("ready",))
        except Exception as ex:      # noqa: BLE001
            self.q.put(("loaderr", str(ex)[:200]))

    def _save_cfg(self):
        for k, var in self.cfg_vars.items():
            self.cfg[k] = var.get().strip() or config.DEFAULTS[k]
        config.save(self.cfg)
        self.name = self.cfg["name"]
        if self.agent is not None:
            self.agent.home = self.cfg["home"]
            self.agent.env.home = self.cfg["home"]
            self.agent.soul = self.cfg["soul"]
            self.agent.new_round()
        self.status.config(text=f"settings saved. place is {self.cfg['home']}.", fg=GOOD)

    def _send(self):
        if self.busy or self.agent is None:
            return
        text = self.input.get().strip()
        if not text:
            return
        self.input.delete(0, "end")
        self._say("you", text)
        self.busy = True
        self.send_btn.config(state="disabled")
        self.neurons.fire(1.2)
        threading.Thread(target=self._run_turn, args=(text,), daemon=True).start()

    def _run_turn(self, text):
        t0 = time.time()
        try:
            r = self.agent.reply(text) if self.pending_ask else self.agent.run(text)
        except Exception as ex:      # noqa: BLE001
            r = {"kind": "fail", "answer": f"error: {str(ex)[:160]}", "trace": []}
        r["seconds"] = time.time() - t0
        self.q.put(("answer", r))

    # ---------------- UI pump
    def _tick(self):
        try:
            while True:
                ev = self.q.get_nowait()
                self._handle(ev)
        except queue.Empty:
            pass
        self.neurons.step()
        now = time.time()
        for label, cells in self.tiles.items():
            lit = self.lit.get(label)
            on = lit and now < lit[0]
            for cell, col in cells:
                cell.config(bg=(lit[1] if on else PANEL2), fg=("#04101f" if on else MUTE))
        self.after(40, self._tick)

    def _flash_tile(self, key):
        label = self.key_label.get(key) or self.key_label.get(ALIAS.get(key, ""))
        cells = self.tiles.get(label, [])
        if cells:
            self.lit[label] = (time.time() + 0.7, cells[0][1])

    def _handle(self, ev):
        kind = ev[0]
        if kind == "ready":
            self.status.config(text="ready. ask it anything below.", fg=GOOD)
            self.input.focus_set()
            auto = os.environ.get("BSLM_SHOWCASE_AUTO")
            if auto:
                self.input.insert(0, auto)
                self.after(400, self._send)
        elif kind == "loaderr":
            self.status.config(text="model did not load:\n" + ev[1] + "\n(is tools/llama/llama-server.exe and the v1 gguf present?)", fg=HOT)
        elif kind == "think":
            self.neurons.fire(0.7)
        elif kind == "act_start":
            name = ev[1].split("(", 1)[0].strip()
            self._flash_tile(name)
            self.neurons.fire(0.6)
            self._logline(ev[1], "act")
            if name in ("lights",):
                self._apply_lights(ev[1])
        elif kind == "act_done":
            self._logline("  " + ev[2].replace("\n", " ")[:80], "res")
        elif kind == "answer":
            r = ev
            self._say("bee", r[1]["answer"])
            n = len(r[1].get("trace", []))
            self._say_dim(f"{n} action(s)  .  {r[1].get('seconds', 0):.1f}s  .  {r[1]['kind']}")
            self.pending_ask = r[1]["kind"] == "ask"
            self.busy = False
            self.send_btn.config(state="normal")
            self.neurons.fire(1.0)
            if os.environ.get("BSLM_SHOWCASE_EXIT"):
                print("AUTO answer:", r[1]["answer"][:100], "| acts", [a for a, _ in r[1].get("trace", [])], flush=True)
                self.after(1500, self.destroy)

    def _apply_lights(self, line):
        # lights("half", "living room")  -> colour the lamps by level
        import re
        m = re.search(r'lights\("([^"]*)"', line)
        lv = (m.group(1) if m else "on").lower()
        levels = {"full": 1.0, "max": 1.0, "bright": 1.0, "high": 0.8, "on": 1.0, "half": 0.5,
                  "medium": 0.5, "low": 0.3, "soft": 0.3, "dim": 0.3, "night": 0.12, "off": 0.0}
        try:
            lvl = int(lv) / 100.0
        except ValueError:
            lvl = levels.get(lv, 0.6)
        base = WARM if lvl > 0 else PANEL2
        a = tuple(int(base[i:i + 2], 16) for i in (1, 3, 5)) if lvl > 0 else (20, 26, 44)
        shade = "#%02x%02x%02x" % tuple(int(20 + (a[i] - 20) * lvl) for i in range(3))
        for oid in self.lamps:
            self.lampcanvas.itemconfig(oid, fill=shade, outline=WARM if lvl > 0 else EDGE)
        self.lightlabel.config(text=f"{lv}  ({int(lvl*100)}%)", fg=WARM if lvl > 0 else MUTE)

    def _say(self, who, text):
        self.chat.config(state="normal")
        self.chat.insert("end", ("You  " if who == "you" else self.name + "  "), who)
        self.chat.insert("end", text + "\n", "msg")
        self.chat.config(state="disabled"); self.chat.see("end")

    def _say_dim(self, text):
        self.chat.config(state="normal")
        self.chat.insert("end", text + "\n", "dim")
        self.chat.config(state="disabled"); self.chat.see("end")

    def _logline(self, text, tag):
        self.log.config(state="normal")
        self.log.insert("end", text + "\n", tag)
        self.log.see("end")
        self.log.config(state="disabled")


def main():
    selftest = "--check" in sys.argv or os.environ.get("BSLM_SHOWCASE_SELFTEST") == "1"
    app = Showcase(selftest=selftest)
    if selftest:
        app.after(1400, app.destroy)
    app.mainloop()


if __name__ == "__main__":
    main()
