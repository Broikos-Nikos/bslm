"""Tiny GUI for BSLM. Remembers the last 30 messages and has a soul.

    python gui.py                       open the window
    python gui.py --test "hello" "..."  run the brain without a window
    python gui.py --send "hello" "..."  type into the OPEN window from a terminal
                                        and print what it answered (testing)
    python gui.py --send /history       the window's current transcript
    python gui.py --send /soul          what it knows about you

memory  data/memory.json   last MAX_MESSAGES messages, survives restarts
soul    data/soul.json     who the assistant is and what it knows about you
control 127.0.0.1:8765     one line in, reply out; local only, testing only
"""
import json
import os
import re
import socket
import sys
import threading
from collections import deque
from datetime import datetime
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent
MEM = ROOT / "data" / "memory.json"
SOUL = ROOT / "data" / "soul.json"
MAX_MESSAGES = 30
PORT = int(os.environ.get("BSLM_GUI_PORT", "8765"))
END = "\n<<end>>\n"

DEFAULT_SOUL = {
    "assistant": {"name": "Bee", "style": "short, warm, never explains itself"},
    "user": {"name": None, "city": None, "likes": [], "facts": []},
}

# things the soul learns directly, before the model sees the text
LEARN = [
    (r"^(?:my name is|call me|the name is)\s+(\S+)\s*$", "name", "set"),   # not "i am tired"
    (r"^με λένε\s+(\S+)\s*$", "name", "set"),
    (r"^(?:i live in|i'm from|i am from)\s+(.+?)\s*$", "city", "set"),
    (r"^μένω\s+(?:στην?|στον?|στο|στα)\s+(.+?)\s*$", "city", "set"),
    (r"^i like\s+(.+?)\s*$", "likes", "add"),
    (r"^μου αρέσ(?:ει|ουν)\s+(.+?)\s*$", "likes", "add"),
    (r"^remember that\s+(.+?)\s*$", "facts", "add"),
    (r"^να θυμάσαι ότι\s+(.+?)\s*$", "facts", "add"),
    (r"^(?:who am i|what do you know about me)\s*\??$", None, "recall"),
    (r"^(?:ποιος είμαι|τι ξέρεις για μένα)\s*;?$", None, "recall"),
]


def load(path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return json.loads(json.dumps(default))


def save(path, obj):
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


class Brain:
    def __init__(self, speak):
        from bslm.infer import Parser
        from bslm.skills import Assistant
        self.soul = load(SOUL, DEFAULT_SOUL)
        self.memory = deque(load(MEM, []), maxlen=MAX_MESSAGES)
        self.parser = Parser()
        self.bot = Assistant(speak=speak)
        self.bot.last = next((m["text"] for m in reversed(self.memory)
                              if m["role"] == "assistant"), "")
        save(SOUL, self.soul)

    # ---- soul -------------------------------------------------------------
    def learn(self, text):
        for pat, field, kind in LEARN:
            m = re.match(pat, text.strip(), re.I)
            if not m:
                continue
            u = self.soul["user"]
            if kind == "recall":
                bits = [f"you are {u['name']}" if u["name"] else "I do not know your name yet"]
                if u["city"]:
                    bits.append(f"you live in {u['city']}")
                if u["likes"]:
                    bits.append("you like " + ", ".join(u["likes"]))
                bits += u["facts"]
                return ". ".join(b[0].upper() + b[1:] for b in bits) + "."
            val = m.group(1).strip(" .!")
            if kind == "set":
                u[field] = val
            elif val not in u[field]:
                u[field].append(val)
            save(SOUL, self.soul)
            return f"Noted, {val}."
        return None

    def apply_soul(self, parsed):
        """Fill missing slots from what the soul knows. A short utterance like
        "play something" lands under the confidence threshold; if the soul can
        complete it, trust the model's top guess."""
        u, s, top = self.soul["user"], parsed["slots"], parsed["raw_intent"]
        if top == "weather.query" and "location" not in s and u["city"]:
            s["location"] = u["city"]
            parsed["intent"] = top
        if top == "music.play" and not s and u["likes"]:
            s["genre"] = u["likes"][0]
            parsed["intent"] = top
        return parsed

    # ---- one exchange -------------------------------------------------------
    def reply(self, text):
        out = self.learn(text)
        if out is None:
            parsed = self.apply_soul(self.parser.parse(text))
            out = self.bot.run(parsed)
            name = self.soul["user"]["name"]
            if parsed["intent"] == "smalltalk.greet" and name:
                out = out.rstrip(".") + f", {name}."
        self.remember("user", text)
        self.remember("assistant", out)
        return out

    def remember(self, role, text):
        self.memory.append({"role": role, "text": text,
                            "at": datetime.now().isoformat(timespec="minutes")})
        save(MEM, list(self.memory))

    def forget(self):
        self.memory.clear()
        save(MEM, [])


def run_window():
    import tkinter as tk
    from tkinter import scrolledtext

    root = tk.Tk()
    log = scrolledtext.ScrolledText(root, width=64, height=22, wrap="word",
                                    font=("Segoe UI", 10), state="disabled")
    log.pack(fill="both", expand=True, padx=8, pady=(8, 4))
    entry = tk.Entry(root, font=("Segoe UI", 11))
    entry.pack(fill="x", padx=8, pady=(0, 8))
    entry.focus()

    def show(role, text):
        log.configure(state="normal")
        log.insert("end", f"{role}: {text}\n")
        log.configure(state="disabled")
        log.see("end")

    brain = Brain(speak=lambda msg: root.after(0, show, "timer", msg.strip()))
    root.title(brain.soul["assistant"]["name"])
    me = brain.soul["assistant"]["name"]
    for m in brain.memory:
        show("you" if m["role"] == "user" else me, m["text"])

    def handle(text):
        """One message, exactly as if typed. Returns what the window showed."""
        if text == "/forget":
            brain.forget()
            log.configure(state="normal")
            log.delete("1.0", "end")
            log.configure(state="disabled")
            return "(memory cleared)"
        if text == "/history":
            return "\n".join(f"{m['role']}: {m['text']}" for m in brain.memory) or "(empty)"
        if text == "/soul":
            return json.dumps(brain.soul, ensure_ascii=False, indent=2)
        show("you", text)
        out = brain.reply(text)
        show(me, out)
        return out

    def send(_=None):
        text = entry.get().strip()
        entry.delete(0, "end")
        if text:
            handle(text)

    def control_server():
        """Testing hook: a terminal can type into this window and read the reply.
        Local only. Each connection: one line in, the reply out, then END."""
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", PORT))
        srv.listen(4)
        while True:
            conn, _ = srv.accept()
            conn.settimeout(10)          # a silent client must not hang the port for everyone
            try:
                text = conn.makefile("r", encoding="utf-8").readline().strip()
                done, box = threading.Event(), []

                def on_main(t=text):
                    try:
                        box.append(handle(t))
                    except Exception as e:                    # noqa: BLE001
                        box.append(f"error: {e}")
                    done.set()

                root.after(0, on_main)
                done.wait(timeout=180)
                conn.sendall(((box[0] if box else "timeout") + END).encode("utf-8"))
            except Exception:
                pass
            finally:
                conn.close()

    threading.Thread(target=control_server, daemon=True).start()
    entry.bind("<Return>", send)
    root.mainloop()


def send_to_window(lines):
    """Terminal client for the control port."""
    for line in lines:
        s = socket.create_connection(("127.0.0.1", PORT), timeout=200)
        s.sendall((line.strip() + "\n").encode("utf-8"))
        buf = b""
        while not buf.decode("utf-8", "replace").endswith(END):
            chunk = s.recv(4096)
            if not chunk:
                break
            buf += chunk
        s.close()
        reply = buf.decode("utf-8", "replace")
        reply = reply[:-len(END)] if reply.endswith(END) else reply
        print(f"> {line}\n  {reply.replace(chr(10), chr(10) + '  ')}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        b = Brain(speak=print)
        for t in sys.argv[2:]:
            print(f"> {t}\n  {b.reply(t)}")
    elif len(sys.argv) > 1 and sys.argv[1] == "--send":
        try:
            send_to_window(sys.argv[2:])
        except ConnectionRefusedError:
            print("no window is open on port", PORT)
    else:
        run_window()
