"""Siri's capability list, and where BSLM stands against each item.

Machine readable so the agent benchmark (OWN_MODEL.md, stage 7) can be built
from the same list. `python -m bslm.scenarios` writes SCENARIOS.md with the
coverage numbers computed, never typed by hand.

Sources: Apple newsroom, June 2026, "Apple introduces Siri AI"; apple.com/siri;
the iPhone User Guide "Use Siri" pages; two community command inventories
(itechguides master list, the extratone gist). Wrappers are excluded on
purpose: voice in and out, wake word, CarPlay, Watch, back to back requests.

Status codes
  R  works today: the 5M router plus skills
  T  router class, the skill is not written yet: small code, no model change
  A  agent: the own model plus a procedure in code (OWN_MODEL.md)
  C  intelligence in scope, needs a device or app connector on the target phone
  X  out of scope: audio fingerprinting, Apple only services, open ended writing
  (camera and vision are not on the list at all, by decision)
"""
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (domain, capability, example request, procedure or tools, status)
SCENARIOS = [
    # ---------------- communication ----------------
    ("Communication", "Call a contact", "call my sister", "contact lookup, dial", "C"),
    ("Communication", "Call back the last caller", "call back the last number", "call log, dial", "C"),
    ("Communication", "Video call", "video call Maria", "contact lookup, video app", "C"),
    ("Communication", "Send a message with content", "text John I am running late", "contact lookup, send", "C"),
    ("Communication", "Read new messages", "read my new messages", "message store, summarise", "C"),
    ("Communication", "Reply to the last message", "reply I will be there at eight", "thread state, send", "C"),
    ("Communication", "Read or search email", "find the hotel confirmation number in my email", "mail search, extract", "C"),
    ("Communication", "Draft and send an email from scratch", "email the accountant that the invoice is sent", "compose from slots, send", "C"),
    ("Communication", "Read notifications, missed calls, voicemail", "any missed calls", "notification store", "C"),
    ("Communication", "Share ETA with a contact", "share my ETA with Anna", "route ETA, send", "C"),
    ("Communication", "Location triggered message", "when I leave work text Pedro I am on my way", "geofence trigger, send", "C"),
    ("Communication", "Live translation during a call", "translate this call", "real time speech pipeline", "X"),
    ("Communication", "Contact details and relationships", "what is Anna's email; Maria is my sister", "contacts, soul", "C"),
    ("Communication", "Contact birthdays", "when is Nick's birthday", "contacts", "C"),

    # ---------------- knowledge ----------------
    ("Knowledge", "Factual question", "why do stars twinkle", "research: search, open 2 to 3 pages, extract", "A"),
    ("Knowledge", "Current fact the model cannot know", "who is the prime minister of Greece", "research", "A"),
    ("Knowledge", "Upcoming event", "when is the next solar eclipse visible from Athens", "research", "A"),
    ("Knowledge", "Follow up question in context", "and where is the best place to see it", "conversation history plus research", "A"),
    ("Knowledge", "Sports scores, standings, player stats", "did Olympiacos win last night", "research", "A"),
    ("Knowledge", "Stock price and market data", "what is Apple trading at", "quote API or research", "A"),
    ("Knowledge", "Define a word", "define serendipity", "dictionary API", "A"),
    ("Knowledge", "Spell a word", "how do you spell necessary", "model", "A"),
    ("Knowledge", "Rhymes, syllables, letter counts", "what rhymes with orange", "dictionary API, code", "A"),
    ("Knowledge", "Translate a phrase", "how do you say thank you in Japanese", "translation API", "C"),
    ("Knowledge", "Unit conversion", "how many miles is 42 kilometers", "calculate", "R"),
    ("Knowledge", "Currency conversion at today's rate", "how much is 200 euros in dollars", "rates fetch, calculate", "A"),
    ("Knowledge", "Arithmetic and percentages", "what is 15 percent of 240", "calculate", "R"),
    ("Knowledge", "Tip and tax totals", "what is 85 plus 24 percent tax", "calculate", "T"),
    ("Knowledge", "Equations, factorials, trig, primes", "solve x squared minus 5x plus 6 equals 0", "symbolic calculator tool", "A"),
    ("Knowledge", "Random number, dice, coin flip", "flip a coin", "code", "T"),
    ("Knowledge", "Geography facts", "what is the capital of Peru", "research or Wikidata", "A"),
    ("Knowledge", "Distance between places", "how far is Berlin from Athens", "geocode, compute", "A"),
    ("Knowledge", "Science reference", "what is the boiling point of ethanol", "research", "A"),
    ("Knowledge", "Trivia and records", "what is the tallest building in the world", "research", "A"),
    ("Knowledge", "Airport codes and flight status", "is flight A3 651 on time", "research", "A"),
    ("Knowledge", "Where to buy a product", "where can I buy a Raspberry Pi 5", "research", "A"),
    ("Knowledge", "Image search", "show me photos of the Acropolis", "open image search", "A"),
    ("Knowledge", "Wikipedia lookup", "tell me about the battle of Salamis", "fetch page, summarise", "A"),
    ("Knowledge", "Open ended brainstorming and creative writing", "brainstorm what to bring to a potluck; write a poem", "open ended generation", "X"),
    ("Knowledge", "Write with Siri: draft, rewrite, proofread anywhere", "rewrite this paragraph in a friendlier tone", "open ended generation", "X"),

    # ---------------- time, alarms, timers, reminders, calendar ----------------
    ("Time", "Current time and date", "what time is it", "clock", "R"),
    ("Time", "Time in another city", "what time is it in Tokyo", "time zone lookup", "A"),
    ("Time", "Days until a date", "how many days until Christmas", "date math", "T"),
    ("Time", "Sunrise and sunset", "when is sunset today", "weather API", "A"),
    ("Timers", "Set, query, cancel a timer", "set a timer for 25 minutes", "timer", "R"),
    ("Timers", "Several named timers", "set a pasta timer for 9 minutes and a tea timer for 3", "named timers", "T"),
    ("Timers", "Pause, resume, reset a timer", "pause the timer", "timer", "T"),
    ("Timers", "Stopwatch", "start a stopwatch", "stopwatch", "T"),
    ("Alarms", "Set an alarm with time, date, label", "wake me at 6:45 tomorrow for the flight", "alarm", "R"),
    ("Alarms", "Modify, delete, enable, disable alarms", "move my alarm to 7; turn off all alarms", "alarm store", "T"),
    ("Alarms", "Query alarms", "what alarms do I have", "alarm store", "T"),
    ("Reminders", "Time based reminder", "remind me to pay the bill on Friday at 9", "reminders", "R"),
    ("Reminders", "Recurring reminder", "remind me every Monday to water the plants", "reminders with recurrence", "T"),
    ("Reminders", "Location based reminder", "remind me to buy milk when I get to the supermarket", "geofence", "C"),
    ("Reminders", "List, filter, complete, delete reminders", "what do I have to do today; delete the dentist reminder", "reminders", "T"),
    ("Lists", "Add items to a named list", "add milk and eggs to the shopping list", "lists", "R"),
    ("Lists", "Read and clear a list", "read my shopping list; clear it", "lists", "T"),
    ("Calendar", "Create an event with title, date, time, place, people", "lunch with Maria on Friday at 1 at Kolonaki", "calendar", "R"),
    ("Calendar", "Query the calendar", "what do I have tomorrow; when is my next meeting; am I free at 3", "calendar", "R"),
    ("Calendar", "Move, rename, cancel an event", "move my 3 pm to 4; change lunch to a coffee chat", "calendar edit", "T"),
    ("Calendar", "Event from a description", "add the dentist appointment from this text", "extract fields, create", "A"),
    ("Calendar", "Proactive: rain before an outdoor event", "(unprompted) rain at 15:00 before your meeting in Kifisia, you commute by scooter, take the car", "scheduler, forecast, soul", "A"),
    ("Calendar", "Proactive: leave now for the next meeting", "(unprompted) leave in 10 minutes to make the 4 pm", "scheduler, route ETA", "A"),

    # ---------------- weather ----------------
    ("Weather", "Current conditions and forecast, here or elsewhere", "what is the weather in Patras tomorrow", "Open-Meteo", "A"),
    ("Weather", "Rain probability, umbrella question", "do I need an umbrella tonight", "Open-Meteo", "A"),
    ("Weather", "Detailed metrics: humidity, wind, UV, pressure", "how windy is it right now", "Open-Meteo", "A"),
    ("Weather", "Weather at a specific hour", "will it rain at 5 pm", "Open-Meteo hourly", "A"),
    ("Weather", "Weekly outlook", "how does the weekend look", "Open-Meteo daily", "A"),

    # ---------------- music, podcasts, media ----------------
    ("Music", "Play a song, artist, album, playlist or genre", "play Hotel California by the Eagles", "library, then YouTube result list, pick, open", "A"),
    ("Music", "Play by lyrics", "play the song that goes we will we will rock you", "search, pick, open", "A"),
    ("Music", "What is playing, who sings this", "who sings this", "player state", "R"),
    ("Music", "Pause, resume, skip, previous, shuffle, repeat", "next song", "player control", "R"),
    ("Music", "Volume", "volume 60; louder", "volume", "R"),
    ("Music", "Like, add to library or playlist", "add this to my workout playlist", "music service", "C"),
    ("Music", "Identify a song by listening", "what song is this", "audio fingerprinting", "X"),
    ("Podcasts", "Play latest or specific episode, subscribe, speed", "play the latest episode of Lex Fridman", "podcast search, open", "A"),
    ("Radio", "Live radio", "play BBC Radio 4", "station search, open stream", "A"),
    ("News", "News briefing", "what is the news today", "news source, summarise", "A"),
    ("Media", "Multi room audio", "play jazz in the kitchen; move the music to the bedroom", "multi room audio system", "C"),
    ("Media", "TV control", "play the next episode of Severance on the TV; pause; subtitles on", "TV API", "C"),

    # ---------------- home ----------------
    ("Home", "Lights by room, brightness, colour", "dim the living room lights to 30 percent", "home connector", "R"),
    ("Home", "Thermostat, heating, AC", "set the temperature to 22", "home connector", "T"),
    ("Home", "Fans, blinds, plugs, devices on and off", "turn on the boiler", "home connector", "R"),
    ("Home", "Scenes", "movie night", "scene definitions", "T"),
    ("Home", "Timed home actions", "turn off all the lights at 11", "scheduler", "A"),
    ("Home", "Locks and garage: status and control", "is the front door locked; lock it", "lock connector", "C"),
    ("Home", "Sensor questions", "what is the temperature in the bedroom", "sensor connector", "C"),
    ("Home", "Several homes", "turn off the lights at the cottage", "multi home connector", "C"),

    # ---------------- navigation and places ----------------
    ("Navigation", "Directions, by driving, walking, transit, bike", "walking directions to the port", "maps", "R"),
    ("Navigation", "Travel time and traffic", "how long to get to the airport right now", "routing API", "A"),
    ("Navigation", "Nearby places and opening hours", "is there a pharmacy open nearby", "places search", "A"),
    ("Navigation", "Fuel and charging stations", "nearest petrol station", "places search", "A"),
    ("Navigation", "Current location, elevation", "where am I", "GPS", "C"),
    ("Navigation", "Stop or change navigation", "stop navigation", "maps connector", "C"),
    ("Navigation", "Where I parked", "where did I park", "location log", "C"),
    ("Navigation", "Report an accident or hazard", "report an accident", "Apple Maps crowdsourcing", "X"),
    ("Navigation", "Locate a friend", "where is Gordon", "Find My", "X"),

    # ---------------- device and system ----------------
    ("Device", "Open an app", "open the camera", "OS", "C"),
    ("Device", "System toggles: Wi-Fi, Bluetooth, flashlight, dark mode, do not disturb, low power", "turn on do not disturb", "OS settings", "C"),
    ("Device", "Brightness and system volume", "brightness 50 percent", "OS settings", "C"),
    ("Device", "Battery level", "what is my battery", "OS", "C"),
    ("Device", "Accessibility toggles", "turn on VoiceOver", "OS", "C"),
    ("Device", "Read a page aloud or summarise it", "read this page; summarise this article", "fetch, summarise", "A"),
    ("Device", "Find my device, ping the headphones", "find my AirPods", "Find My", "X"),
    ("Device", "Passwords", "show my Netflix password", "password manager", "X"),
    ("Device", "Wallet, Apple Pay, split a bill", "split the bill with Alex", "Apple Pay", "X"),
    ("Device", "Send money to a contact", "send 20 euros to Nick", "payments", "X"),

    # ---------------- personal context and memory ----------------
    ("Personal", "Find a thing a friend messaged", "find the restaurant Anna recommended", "personal index over messages", "C"),
    ("Personal", "Surface a number from an old email", "what is my hotel confirmation number", "personal index over mail", "C"),
    ("Personal", "On screen awareness: ask about or act on what is on screen", "add this recipe to my notes", "screen text access", "C"),
    ("Personal", "Remember user facts and preferences", "I commute by scooter; call me Nick", "soul", "R"),
    ("Personal", "Conversation memory and follow ups", "what did I ask you before", "memory", "R"),
    ("Personal", "Personal routines and macros", "when I say good night, turn off the lights and set the alarm", "macro store", "A"),
    ("Personal", "Third party app actions", "order my usual from Wolt; log a run in Strava", "app connectors", "C"),

    # ---------------- notes and web ----------------
    ("Notes", "Create, read, search notes", "note that the wifi password is hunter2", "notes", "R"),
    ("Notes", "Save a recipe or article to notes", "save this recipe", "fetch, extract, note", "A"),
    ("Web", "Web search", "search for cheap flights to Berlin", "open search", "R"),
    ("Web", "Price or availability watch", "tell me when this drops below 300 euros", "scheduler, fetch, compare", "A"),
    ("Web", "Summarise a web page", "summarise this page", "fetch, summarise", "A"),

    # ---------------- health and fitness ----------------
    ("Health", "Log medication, water, weight", "log that I took my vitamins", "health store", "C"),
    ("Health", "Read health data: steps, sleep, heart rate", "how did I sleep", "health store", "C"),
    ("Health", "Workout coaching", "start a run with voice coaching", "workout buddy", "X"),


    # ---------------- small talk ----------------
    ("Small talk", "Greetings, thanks, jokes, feelings", "tell me a joke; I am bored", "canned and templated", "T"),
    ("Small talk", "Games", "rock paper scissors", "code", "T"),
    ("Small talk", "Capabilities and help", "what can you do", "canned", "R"),
]


def coverage():
    c = Counter(s[4] for s in SCENARIOS)
    n = len(SCENARIOS)
    return n, c


def main():
    n, c = coverage()
    matched = c["R"] + c["T"] + c["A"] + c["C"]
    own = c["R"] + c["T"] + c["A"]
    by_domain = {}
    for d, *_ , st in SCENARIOS:
        by_domain.setdefault(d, Counter())[st] += 1

    rows = ["| # | domain | Siri capability | example | procedure or tools | status |",
            "|---|---|---|---|---|---|"]
    for i, (d, cap, ex, proc, st) in enumerate(SCENARIOS, 1):
        rows.append(f"| {i} | {d} | {cap} | {ex} | {proc} | **{st}** |")

    dom_rows = ["| domain | items | R | T | A | C | X | matched |", "|---|---|---|---|---|---|---|---|"]
    for d, cnt in by_domain.items():
        t = sum(cnt.values())
        m = cnt["R"] + cnt["T"] + cnt["A"] + cnt["C"]
        dom_rows.append(f"| {d} | {t} | {cnt['R']} | {cnt['T']} | {cnt['A']} | {cnt['C']} | {cnt['X']} | {100*m/t:.0f}% |")

    md = f"""# Siri capability match

Generated by `python -m bslm.scenarios`. {n} capabilities from Apple's own
material (the June 2026 Siri AI announcement, apple.com/siri, the iPhone
User Guide) plus two community command inventories. Wrappers are excluded on
purpose: voice in and out, wake word, CarPlay, Watch, back to back requests.

## The bar

Two numbers, both must clear 80%, and 79% is a fail, not a rounding error:

1. **Coverage.** Share of Siri's list whose *intelligence* BSLM implements.
2. **Accuracy.** On the agent benchmark built from this list, every domain
   must pass at least 80% of its cases. A domain at 79% blocks the release.

## Coverage today and planned

| | count | share |
|---|---|---|
| **R** works today, router plus skills | {c['R']} | {100*c['R']/n:.1f}% |
| **T** router class, skill not written yet, no model change | {c['T']} | {100*c['T']/n:.1f}% |
| **A** agent: own model plus procedure in code | {c['A']} | {100*c['A']/n:.1f}% |
| **C** intelligence in scope, needs a phone connector | {c['C']} | {100*c['C']/n:.1f}% |
| **X** out of scope | {c['X']} | {100*c['X']/n:.1f}% |
| **matched on intelligence (R + T + A + C)** | {matched} | **{100*matched/n:.1f}%** |
| works end to end with no device connector (R + T + A) | {own} | {100*own/n:.1f}% |

Camera and vision items are not on the list at all, by decision. What **X**
contains, so the exclusions are visible: audio fingerprinting, Apple only
services (Find My, Apple Pay, Wallet, passwords, Maps crowdsourcing), real
time call translation, workout coaching, and open ended writing
(brainstorming, drafting, rewriting). The last one is
a deliberate choice: a 100 to 300M model trained on procedures is not a
writer, and pretending otherwise is how you end up at 79%.

## By domain

{chr(10).join(dom_rows)}

## The list

{chr(10).join(rows)}

## How this becomes the benchmark

Every **R**, **T** and **A** row gets at least ten test cases (different phrasings,
different slot values, English only) with an expected tool trace and an
expected answer. **C** rows are tested against a mock connector so the
intelligence is measured even before a phone integration exists. The suite
lives next to `bslm/benchmark.py`, and the per domain pass rate is the number
that decides whether a model size ships (OWN_MODEL.md, stage 7).
"""
    out = ROOT / "SCENARIOS.md"
    out.write_text(md, encoding="utf-8", newline="\n")
    print(f"{n} capabilities: R {c['R']}  T {c['T']}  A {c['A']}  C {c['C']}  X {c['X']}  "
          f"matched {100*matched/n:.1f}%  own only {100*own/n:.1f}%")
    print(f"written {out}")


if __name__ == "__main__":
    main()
