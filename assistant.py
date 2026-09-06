"""BSLM assistant. Usage:

    python assistant.py                     interactive
    python assistant.py "set a timer for 5 minutes"
    python assistant.py --debug             show intent, confidence and slots
"""
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from bslm.infer import Parser          # noqa: E402
from bslm.skills import Assistant      # noqa: E402


def main():
    args = [a for a in sys.argv[1:]]
    debug = "--debug" in args
    args = [a for a in args if a != "--debug"]

    parser = Parser()
    bot = Assistant()

    def handle(text):
        p = parser.parse(text)
        if debug:
            slots = ", ".join(f"{k}={v!r}" for k, v in p["slots"].items()) or "none"
            print(f"  intent {p['raw_intent']} ({p['confidence']:.3f})  slots: {slots}")
        print(bot.run(p))

    if args:
        handle(" ".join(args))
        return

    print("BSLM ready. English or Greek. Ctrl+C to quit.")
    while True:
        try:
            text = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return
        if not text:
            continue
        if text in {"exit", "quit"}:
            return
        handle(text)


if __name__ == "__main__":
    main()
