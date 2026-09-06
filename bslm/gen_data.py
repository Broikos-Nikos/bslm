"""Synthetic corpus generator for the BSLM assistant.

Produces utterances with an intent label and character level slot spans.
Bilingual: English and Greek. No pretrained anything, the data is the model.
"""
import json
import random
import re
import unicodedata
from pathlib import Path

RNG = random.Random(20260903)

# --------------------------------------------------------------------------
# slot value pools
# --------------------------------------------------------------------------

EN_NUM = {
    1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
    8: "eight", 9: "nine", 10: "ten", 15: "fifteen", 20: "twenty", 30: "thirty",
    45: "forty five", 60: "sixty", 90: "ninety",
}
EL_NUM = {
    1: "ένα", 2: "δύο", 3: "τρία", 4: "τέσσερα", 5: "πέντε", 6: "έξι", 7: "εφτά",
    8: "οκτώ", 9: "εννιά", 10: "δέκα", 15: "δεκαπέντε", 20: "είκοσι",
    30: "τριάντα", 45: "σαράντα πέντε", 60: "εξήντα", 90: "ενενήντα",
}


def durations(lang):
    out = []
    for n in [1, 2, 3, 5, 7, 10, 12, 15, 20, 25, 30, 40, 45, 60, 90]:
        if lang == "en":
            units = ["seconds", "minutes", "hours"] if n > 1 else ["second", "minute", "hour"]
            for u in units:
                out.append(f"{n} {u}")
                if n in EN_NUM:
                    out.append(f"{EN_NUM[n]} {u}")
            out.append(f"{n} min")
        else:
            for u in ["δευτερόλεπτα", "λεπτά", "ώρες"]:
                out.append(f"{n} {u}")
                if n in EL_NUM:
                    out.append(f"{EL_NUM[n]} {u}")
            out.append(f"{n} λεπτο" if n == 1 else f"{n} λεπτά")
    out += ["half an hour", "an hour and a half", "a minute"] if lang == "en" else \
           ["μισή ώρα", "μιάμιση ώρα", "ένα λεπτό", "ενάμιση λεπτό"]
    return out


def times(lang):
    out = []
    for h in range(1, 13):
        for m in ["00", "15", "30", "45"]:
            out.append(f"{h}:{m}")
    for h in range(6, 24):
        out.append(f"{h:02d}:00")
        out.append(f"{h:02d}:30")
    if lang == "en":
        for h in range(1, 13):
            out += [f"{h} am", f"{h} pm", f"{h} o'clock", f"half past {h}",
                    f"quarter past {h}", f"quarter to {h}"]
        out += ["noon", "midnight", "sunrise", "sunset"]
    else:
        for h in range(1, 13):
            out += [f"στις {h}", f"{h} το πρωί", f"{h} το απόγευμα", f"{h} το βράδυ",
                    f"{h} και μισή", f"{h} και τέταρτο", f"{h} παρά τέταρτο"]
        out += ["το μεσημέρι", "τα μεσάνυχτα", "το ξημέρωμα"]
    return out


def dates(lang):
    if lang == "en":
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        months = ["January", "February", "March", "April", "May", "June", "July",
                  "August", "September", "October", "November", "December"]
        out = ["today", "tomorrow", "tonight", "this evening", "this afternoon",
               "the day after tomorrow", "next week", "this weekend", "next month"]
        for d in days:
            out += [d, f"next {d}", f"this {d}", f"on {d}"]
        for m in months:
            for n in [1, 3, 5, 8, 12, 15, 18, 21, 24, 27, 30]:
                out.append(f"{m} {n}")
        return out
    days = ["Δευτέρα", "Τρίτη", "Τετάρτη", "Πέμπτη", "Παρασκευή", "Σάββατο", "Κυριακή"]
    months = ["Ιανουαρίου", "Φεβρουαρίου", "Μαρτίου", "Απριλίου", "Μαΐου", "Ιουνίου",
              "Ιουλίου", "Αυγούστου", "Σεπτεμβρίου", "Οκτωβρίου", "Νοεμβρίου", "Δεκεμβρίου"]
    out = ["σήμερα", "αύριο", "απόψε", "μεθαύριο", "το βράδυ", "το απόγευμα",
           "την επόμενη εβδομάδα", "το σαββατοκύριακο", "τον επόμενο μήνα"]
    for d in days:
        out += [f"την {d}", f"την επόμενη {d}", d]
    for m in months:
        for n in [1, 3, 5, 8, 12, 15, 18, 21, 24, 27, 30]:
            out.append(f"{n} {m}")
    return out


PERSON = {
    "en": ["John", "Maria", "Alex", "Sarah", "Michael", "Anna", "David", "Elena",
           "Chris", "Sophia", "Tom", "Nick", "Kate", "George", "Lisa", "Peter",
           "mom", "dad", "my brother", "my sister", "my boss", "the office",
           "Dr Smith", "James Miller", "Emily Watson", "my wife", "my landlord",
           "the accountant", "the lawyer", "the plumber", "the client", "the bank",
           "the doctor", "the electrician", "the recruiter", "the supplier", "my brother"],
    "el": ["τον Γιώργο", "τη Μαρία", "τον Νίκο", "την Ελένη", "τον Κώστα", "την Άννα",
           "τον Δημήτρη", "τη Σοφία", "τον Παναγιώτη", "την Κατερίνα", "τη μαμά",
           "τον μπαμπά", "τον αδερφό μου", "την αδερφή μου", "το αφεντικό",
           "τον κύριο Παπαδόπουλο", "τη γυναίκα μου", "τον λογιστή"],
}

LOCATION = {
    "en": ["Athens", "Thessaloniki", "London", "Berlin", "Paris", "New York",
           "Rome", "Madrid", "Amsterdam", "Tokyo", "Crete", "Santorini",
           "Patras", "Larissa", "Volos", "Heraklion", "here", "the airport"],
    "el": ["στην Αθήνα", "στη Θεσσαλονίκη", "στο Λονδίνο", "στο Βερολίνο",
           "στο Παρίσι", "στη Νέα Υόρκη", "στη Ρώμη", "στη Μαδρίτη", "στην Κρήτη",
           "στη Σαντορίνη", "στην Πάτρα", "στη Λάρισα", "στον Βόλο", "στο Ηράκλειο",
           "στα Ιωάννινα", "στη Ρόδο", "εδώ"],
}

ROOM = {
    "en": ["kitchen", "bedroom", "living room", "bathroom", "office", "hallway",
           "garage", "balcony", "dining room", "kids room"],
    "el": ["κουζίνα", "υπνοδωμάτιο", "σαλόνι", "μπάνιο", "γραφείο", "διάδρομο",
           "γκαράζ", "μπαλκόνι", "τραπεζαρία", "παιδικό δωμάτιο"],
}

SONG = {
    "en": ["Bohemian Rhapsody", "Smells Like Teen Spirit", "Hotel California",
           "Billie Jean", "Wonderwall", "Blinding Lights", "Yesterday",
           "Stairway to Heaven", "Take Five", "Purple Rain", "Losing My Religion",
           "Sweet Child O Mine", "Clocks", "Zombie", "Africa"],
    "el": ["Το τρένο φεύγει στις οκτώ", "Μάτια βουρκωμένα", "Αχ Ελλάδα σ' αγαπώ",
           "Ένα το χελιδόνι", "Πάμε στοίχημα", "Χίλιες σιωπές", "Άσε με",
           "Ο ταξιτζής", "Ζήλεια", "Μη μου θυμώνεις μάτια μου"],
}

ARTIST = {
    "en": ["Queen", "Nirvana", "The Beatles", "Radiohead", "Coldplay", "Adele",
           "Miles Davis", "Pink Floyd", "Daft Punk", "Kendrick Lamar", "Bjork",
           "Fleetwood Mac", "The Weeknd", "Arctic Monkeys"],
    "el": ["τον Σαββόπουλο", "τον Χατζιδάκι", "τον Θεοδωράκη", "την Αλκηστη Πρωτοψάλτη",
           "τους Πυξ Λαξ", "τον Μαχαιρίτσα", "τη Βίσση", "τον Νταλάρα",
           "τους Τρύπες", "τον Παπακωνσταντίνου"],
}

PLAYLIST = {
    "en": ["my workout playlist", "chill vibes", "focus mix", "road trip",
           "discover weekly", "the driving playlist", "sunday morning", "my liked songs"],
    "el": ["τη λίστα για γυμναστήριο", "τα αγαπημένα μου", "τη λίστα για δουλειά",
           "τη λίστα για το αυτοκίνητο", "χαλαρωτικά", "ελληνικά"],
}

GENRE = {
    "en": ["jazz", "rock", "classical music", "hip hop", "techno", "reggae",
           "lo fi", "metal", "pop", "blues"],
    "el": ["ροκ", "τζαζ", "κλασική μουσική", "ρεμπέτικα", "λαϊκά", "έντεχνα",
           "ξένη μουσική", "παραδοσιακά"],
}

ITEM = {
    "en": ["milk", "bread", "eggs", "coffee", "olive oil", "tomatoes", "rice",
           "chicken", "batteries", "toothpaste", "dog food", "sugar", "yogurt",
           "printer paper", "light bulbs", "washing powder"],
    "el": ["γάλα", "ψωμί", "αυγά", "καφέ", "ελαιόλαδο", "ντομάτες", "ρύζι",
           "κοτόπουλο", "μπαταρίες", "οδοντόκρεμα", "ζάχαρη", "γιαούρτι",
           "χαρτί κουζίνας", "λάμπες", "απορρυπαντικό"],
}

LIST_NAME = {
    "en": ["shopping list", "grocery list", "todo list", "packing list", "wish list"],
    "el": ["λίστα με τα ψώνια", "λίστα του σούπερ μάρκετ", "λίστα με τις δουλειές",
           "λίστα ταξιδιού"],
}

TASK = {
    "en": ["call the bank", "pay the electricity bill", "water the plants",
           "book the flight tickets", "send the invoice", "renew my passport",
           "buy a gift for Anna", "take the car for service", "back up the laptop",
           "review the contract", "pick up the kids", "go to the gym",
           "email the accountant", "cancel the subscription", "print the tickets",
           "renew the domain", "renew the ssl", "call the accountant", "send the quote",
           "pay the vat", "answer the recruiter", "check the pi", "restart the bot",
           "update the plugin", "post on linkedin", "buy dog food", "book the mot"],
    "el": ["να πάρω την τράπεζα", "να πληρώσω το ρεύμα", "να ποτίσω τα φυτά",
           "να κλείσω εισιτήρια", "να στείλω το τιμολόγιο", "να ανανεώσω το διαβατήριο",
           "να πάω στο σέρβις", "να ελέγξω το συμβόλαιο", "να πάρω τα παιδιά",
           "να πάω γυμναστήριο", "να στείλω μέιλ στον λογιστή", "να κόψω τη συνδρομή"],
}

EVENT = {
    "en": ["team meeting", "dentist appointment", "lunch with Maria", "standup",
           "client call", "birthday party", "gym session", "flight to Berlin",
           "quarterly review", "coffee with Alex", "parents evening", "interview"],
    "el": ["σύσκεψη", "ραντεβού στον οδοντίατρο", "γεύμα με τη Μαρία", "τηλεδιάσκεψη",
           "κλήση με πελάτη", "γενέθλια", "προπόνηση", "πτήση για Βερολίνο",
           "ραντεβού στον γιατρό", "καφέ με τον Νίκο"],
}

MESSAGE = {
    "en": ["I am running late", "see you in ten minutes", "happy birthday",
           "the meeting is moved to Friday", "can you call me back",
           "I will be there at eight", "thanks for today", "the invoice is sent",
           "we need to talk about the contract", "on my way"],
    "el": ["θα αργήσω λίγο", "τα λέμε σε δέκα λεπτά", "χρόνια πολλά",
           "η σύσκεψη πάει Παρασκευή", "πάρε με τηλέφωνο", "θα είμαι εκεί στις οκτώ",
           "ευχαριστώ για σήμερα", "έστειλα το τιμολόγιο", "έρχομαι"],
}

NOTE = {
    "en": ["the wifi password is hunter2", "parking spot B14",
           "the client prefers email over calls", "server reboots every Sunday",
           "Anna likes white wine", "the spare key is under the pot",
           "invoice number 4471", "meeting room is on the third floor"],
    "el": ["ο κωδικός του wifi είναι 12345", "θέση πάρκινγκ Β14",
           "ο πελάτης προτιμά μέιλ", "το κλειδί είναι κάτω από τη γλάστρα",
           "τιμολόγιο 4471", "η αίθουσα είναι στον τρίτο όροφο"],
}

TOPIC = {
    "en": ["technology", "politics", "sports", "the economy", "football",
           "artificial intelligence", "the weather", "Greece", "the stock market"],
    "el": ["τεχνολογία", "πολιτική", "αθλητικά", "οικονομία", "ποδόσφαιρο",
           "τεχνητή νοημοσύνη", "Ελλάδα", "χρηματιστήριο"],
}

LANGUAGE = {
    "en": ["Greek", "English", "German", "French", "Spanish", "Italian",
           "Japanese", "Turkish", "Russian", "Portuguese"],
    "el": ["ελληνικά", "αγγλικά", "γερμανικά", "γαλλικά", "ισπανικά", "ιταλικά",
           "ιαπωνικά", "τούρκικα", "ρώσικα"],
}

PHRASE = {
    "en": ["good morning", "thank you very much", "where is the train station",
           "how much does it cost", "I would like a coffee", "see you tomorrow",
           "the bill please", "I do not understand"],
    "el": ["καλημέρα", "ευχαριστώ πολύ", "πού είναι ο σταθμός", "πόσο κάνει",
           "θα ήθελα έναν καφέ", "τα λέμε αύριο", "τον λογαριασμό παρακαλώ"],
}

DEVICE = {
    "en": ["tv", "television", "air conditioner", "heater", "fan", "coffee machine",
           "washing machine", "printer", "speakers", "oven", "boiler"],
    "el": ["τηλεόραση", "κλιματιστικό", "θερμοσίφωνα", "ανεμιστήρα", "καφετιέρα",
           "πλυντήριο", "εκτυπωτή", "ηχεία", "φούρνο"],
}

COLOR = {
    "en": ["red", "blue", "green", "warm white", "cold white", "purple", "orange", "pink"],
    "el": ["κόκκινο", "μπλε", "πράσινο", "θερμό λευκό", "ψυχρό λευκό", "μωβ", "πορτοκαλί"],
}

UNIT_PAIRS = [
    ("kilometers", "miles"), ("miles", "kilometers"), ("kilos", "pounds"),
    ("pounds", "kilos"), ("celsius", "fahrenheit"), ("fahrenheit", "celsius"),
    ("euros", "dollars"), ("dollars", "euros"), ("meters", "feet"),
    ("feet", "meters"), ("liters", "gallons"), ("inches", "centimeters"),
    ("km", "miles"), ("miles", "km"), ("kg", "lbs"), ("lbs", "kg"), ("kilos", "lbs"),
    ("euros", "usd"), ("eur", "usd"), ("usd", "eur"), ("c", "f"), ("f", "c"),
]
UNIT_PAIRS_EL = [
    ("χιλιόμετρα", "μίλια"), ("μίλια", "χιλιόμετρα"), ("κιλά", "λίβρες"),
    ("λίβρες", "κιλά"), ("κελσίου", "φαρενάιτ"), ("ευρώ", "δολάρια"),
    ("δολάρια", "ευρώ"), ("μέτρα", "πόδια"), ("λίτρα", "γαλόνια"),
]

QUERY = {
    "en": ["the best pizza in Athens", "how to fix a leaking tap",
           "python list comprehension", "flights to Berlin in October",
           "who won the game last night", "opening hours of the post office",
           "cheap hotels in Rome", "how tall is the Eiffel tower"],
    "el": ["καλύτερη πίτσα στην Αθήνα", "πώς φτιάχνω μουσακά",
           "ωράριο ταχυδρομείου", "φθηνά ξενοδοχεία στη Ρώμη",
           "ποιος κέρδισε χθες", "καιρός για ψάρεμα"],
}

MATH = []
for _ in range(400):
    a, b = RNG.randint(2, 999), RNG.randint(2, 99)
    op_en = RNG.choice(["plus", "minus", "times", "divided by", "+", "-", "*", "/"])
    MATH.append(f"{a} {op_en} {b}")
for _ in range(120):
    a = RNG.randint(2, 90)
    MATH.append(f"{a} percent of {RNG.randint(20, 5000)}")
MATH_EL = []
for _ in range(300):
    a, b = RNG.randint(2, 999), RNG.randint(2, 99)
    MATH_EL.append(f"{a} {RNG.choice(['συν', 'πλην', 'επί', 'δια', '+', '-', '*', '/'])} {b}")
for _ in range(80):
    MATH_EL.append(f"{RNG.randint(2, 90)} τοις εκατό του {RNG.randint(20, 5000)}")

LEVEL = [str(n) for n in range(0, 101, 5)] + ["max", "maximum", "half", "full"]
LEVEL_EL = [str(n) for n in range(0, 101, 5)] + ["μέγιστη", "τέρμα", "στη μέση"]

POOLS = {
    "duration": durations,
    "time": times,
    "date": dates,
}
TARGET = {
    "en": ["the checkout page", "the login form", "the header", "the footer", "the menu",
           "the search", "the cart", "the contact form", "the trade log", "the dashboard",
           "the signals", "the tokenizer", "the training loop", "the scraper", "the bot",
           "the landing page", "the product page", "the api", "the database", "the backup",
           "the cron job", "the email sending", "the pdf export", "the image upload",
           "the mobile layout", "the greek translation", "the pricing table", "the sitemap",
           "the temp folder", "the downloads folder", "the old installers", "the msi cache",
           "the exporter", "the raspberry", "the store", "the site", "the pi",
           "the logs", "the numbers", "the deploy", "the css", "the plugin", "the model",
           "the whole project", "everything", "the disk", "drive c", "the pc", "the games",
           "the project", "the training", "the tests", "the release", "the migration",
           "the invoice module", "the export", "the import", "the cache", "the queue",
           "the worker", "the scheduler", "the notifications", "the login", "the signup",
           "the payment flow", "the equity numbers", "the holdings", "the trade fees",
           "the bot on the pi", "the other pc", "the backup disk", "the desktop"],
    "el": ["τη φόρμα", "το μενού", "το καλάθι", "το site", "τη βάση", "τα logs", "το bot",
           "την αρχική", "τη μετάφραση", "τον εκτυπωτή", "το backup"],
}
PROJECT = {
    "en": ["store", "saas", "portfolio", "bot", "energy platform", "booking app", "client site",
           "landing", "exporter", "training pipeline", "blog", "shop"],
    "el": ["site", "eshop", "bot", "πλατφόρμα", "εφαρμογή"],
}
WORK_TOPIC = {
    "en": ["rag", "fine tuning", "small language models", "stripe fees", "viva fees",
           "the demo environment", "the stop loss", "vps hosting", "cloudflare", "a 5 page site",
           "the booking platform", "the events platform", "seo for the store", "google ads",
           "the domain transfer", "the ssl renewal", "the pricing", "the launch",
           "a xeon cpu", "the new server", "the gpu", "python vs php for this",
           "the wordpress plugin", "the job offer", "the certification", "the contract",
           "phase a", "the proposal", "the redesign", "the newsletter",
           "the domain under his name", "a discount", "the source code", "a refund",
           "to pay later", "a split invoice", "the hosting bill", "the yearly contract"],
    "el": ["το seo", "τα google ads", "την προσφορά", "το hosting", "την τιμολόγηση", "το redesign"],
}
UNKNOWN_TOPIC = {
    "en": ["squirreltemp", "zentrix", "the fooberry thing", "kvothe", "ms-playwright", "avg-g3",
           "cryptoquant", "the binance demo", "wsl2", "ollama", "bluestacks", "the openclaw folder",
           "hermes", "vllm", "quantization aware training", "the pnl", "unrealised pnl",
           "the ultimate oscillator", "a xeon gold 6242", "ddr5 vs ddr4", "pcie 5", "the m2 slot",
           "atom editor", "squirrel updater", "the crashdumps", "uv cache", "pip cache",
           "windows terminal", "the msi cache", "cloudflare tunnel", "tailscale", "the rpi zero"],
    "el": ["το squirreltemp", "το ollama", "το bluestacks", "το vllm"],
}
FACT = {
    "en": ["commute by scooter", "live in the city centre", "work from home", "prefer email",
           "dont drink coffee", "use windows", "have two monitors", "wake up at 7",
           "hate small talk", "am a developer", "have a client meeting every monday",
           "drive an old car", "run a small agency", "am left handed", "eat late",
           "have a dog", "speak greek and english", "take the bike to work"],
    "el": ["πηγαίνω με πατίνι", "μένω στην αθήνα", "δουλεύω από το σπίτι"],
}
RULE = {
    "en": ["use dashes", "question me", "guess, search first", "touch the raspberry",
           "be terse", "answer in one line", "suggest text to speech", "use emojis",
           "ask before deleting", "keep the process documented", "use english only",
           "run anything on the gpu without asking", "revert my changes", "use a pretrained model",
           "speak greek", "explain yourself", "add features i did not ask for", "rent compute",
           "check the log before answering", "write the summary at the end", "keep it under 80 mb",
           "commit without asking", "open a browser window", "send emails as the site",
           "suggesting microphones", "suggesting text to speech", "suggesting wrappers",
           "adding emojis", "asking questions", "changing the plan", "apologising",
           "repeating yourself", "guessing", "summarising at the end", "using bullet points"],
    "el": ["χρησιμοποιείς παύλες", "με ρωτάς", "μαντεύεις", "αγγίζεις το raspberry", "είσαι σύντομος"],
}
STATIC_POOLS = {
    "target": TARGET, "project": PROJECT, "rule": RULE, "fact": FACT,
    "person": PERSON, "location": LOCATION, "room": ROOM, "song": SONG,
    "artist": ARTIST, "playlist": PLAYLIST, "genre": GENRE, "item": ITEM,
    "list_name": LIST_NAME, "task": TASK, "title": EVENT, "content": MESSAGE,
    "note": NOTE, "language": LANGUAGE, "phrase": PHRASE,
    "topic": {"en": TOPIC["en"] + WORK_TOPIC["en"] + UNKNOWN_TOPIC["en"],
              "el": TOPIC["el"] + WORK_TOPIC["el"] + UNKNOWN_TOPIC["el"]},
    "device": DEVICE, "color": COLOR, "query": QUERY,
}


def pool(slot, lang):
    if slot in POOLS:
        return POOLS[slot](lang)
    if slot in STATIC_POOLS:
        return STATIC_POOLS[slot][lang]
    if slot == "expression":
        return MATH if lang == "en" else MATH_EL
    if slot == "level":
        return LEVEL if lang == "en" else LEVEL_EL
    if slot == "brightness":
        return [str(n) for n in range(10, 101, 10)] + (
            ["max", "dim"] if lang == "en" else ["μέγιστη", "χαμηλά"])
    if slot == "amount":
        return [str(RNG.randint(1, 500)) for _ in range(200)] + \
               ["2.5", "1.5", "0.5", "12.7", "100", "250"]
    raise KeyError(slot)


# --------------------------------------------------------------------------
# templates: {slot} placeholders, one list per language
# --------------------------------------------------------------------------

from .templates import LEX, OOS, TEMPLATES  # noqa: E402

# fillers that get prefixed or suffixed to any command
PREFIX = {
    "en": ["", "", "", "", "hey", "ok", "please", "can you", "could you",
           "I want you to", "would you", "hey assistant", "listen", "so",
           "quick one", "right", "alright", "do me a favour and"],
    "el": ["", "", "", "", "\u03ad\u03bb\u03b1", "\u03c9\u03c1\u03b1\u03af\u03b1",
           "\u03c0\u03b1\u03c1\u03b1\u03ba\u03b1\u03bb\u03ce",
           "\u03bc\u03c0\u03bf\u03c1\u03b5\u03af\u03c2 \u03bd\u03b1",
           "\u03b8\u03b1 \u03bc\u03c0\u03bf\u03c1\u03bf\u03cd\u03c3\u03b5\u03c2 \u03bd\u03b1",
           "\u03b8\u03ad\u03bb\u03c9 \u03bd\u03b1", "\u03ac\u03ba\u03bf\u03c5",
           "\u03bb\u03bf\u03b9\u03c0\u03cc\u03bd", "\u03b3\u03b9\u03b1",
           "\u03ad\u03bb\u03b1 \u03c1\u03b5",
           "\u03ba\u03ac\u03bd\u03b5 \u03bc\u03bf\u03c5 \u03c4\u03b7 \u03c7\u03ac\u03c1\u03b7 \u03ba\u03b1\u03b9"],
}
SUFFIX = {
    "en": ["", "", "", "", "", "please", "thanks", "for me", "now", "ok?",
           "cheers", "will you", "if you can"],
    "el": ["", "", "", "", "", "\u03c0\u03b1\u03c1\u03b1\u03ba\u03b1\u03bb\u03ce",
           "\u03c3\u03b5 \u03c0\u03b1\u03c1\u03b1\u03ba\u03b1\u03bb\u03ce",
           "\u03c4\u03ce\u03c1\u03b1", "\u03b5\u03bd\u03c4\u03ac\u03be\u03b5\u03b9;",
           "\u03b1\u03bd \u03bc\u03c0\u03bf\u03c1\u03b5\u03af\u03c2",
           "\u03c1\u03b5 \u03c3\u03c5"],
}

from .tokenizer import WORD_RE  # noqa: E402  one pre-tokenizer for data and inference

PLACEHOLDER = re.compile(r"\{(\w+)\}")
LEXMARK = re.compile(r"\[(\w+)\]")


def split_lex(frac=0.25):
    """Hold out a quarter of every synonym group, the test set only sees those."""
    train, test = {}, {}
    for key, langs in LEX.items():
        train[key], test[key] = {}, {}
        for lang, words in langs.items():
            n = max(1, int(len(words) * frac))
            idx = set(RNG.sample(range(len(words)), n))
            test[key][lang] = [w for i, w in enumerate(words) if i in idx]
            train[key][lang] = [w for i, w in enumerate(words) if i not in idx]
    return train, test


def apply_lex(tpl, lang, lex):
    return LEXMARK.sub(lambda m: RNG.choice(lex[m.group(1)][lang]), tpl)


def typo(word):
    if len(word) < 4:
        return word
    i = RNG.randrange(1, len(word) - 1)
    mode = RNG.random()
    if mode < 0.4:
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    if mode < 0.7:
        return word[:i] + word[i + 1:]
    return word[:i] + word[i] + word[i:]


def render(intent, lang, lex, tpl=None):
    """Return (text, [(start, end, slot)])."""
    tpl = tpl if tpl is not None else RNG.choice(TEMPLATES[intent][lang])
    tpl = apply_lex(tpl, lang, lex)
    if intent == "unit.convert":
        uf, ut = RNG.choice(UNIT_PAIRS if lang == "en" else UNIT_PAIRS_EL)
    parts, spans, cursor, pos = [], [], 0, 0
    seen_items = set()
    for m in PLACEHOLDER.finditer(tpl):
        parts.append(tpl[pos:m.start()])
        cursor += len(tpl[pos:m.start()])
        slot = m.group(1)
        if slot == "unit_from":
            val = uf
        elif slot == "unit_to":
            val = ut
        else:
            choices = pool(slot, lang)
            val = RNG.choice(choices)
            if slot == "item":
                for _ in range(6):
                    if val not in seen_items:
                        break
                    val = RNG.choice(choices)
                seen_items.add(val)
        spans.append((cursor, cursor + len(val), slot))
        parts.append(val)
        cursor += len(val)
        pos = m.end()
    parts.append(tpl[pos:])
    text = "".join(parts)

    if intent.startswith(("smalltalk.", "assistant.")):
        pre = RNG.choice(["", "", "", "hey", "ok"] if lang == "en"
                         else ["", "", "", "\u03ad\u03bb\u03b1", "\u03c9\u03c1\u03b1\u03af\u03b1"])
        suf = ""
    else:
        pre = RNG.choice(PREFIX[lang])
        suf = RNG.choice(SUFFIX[lang])
    if pre:
        text = pre + " " + text
        spans = [(s + len(pre) + 1, e + len(pre) + 1, t) for s, e, t in spans]
    if suf:
        text = text + " " + suf
    if RNG.random() < 0.10:
        text = text + RNG.choice(["?", ".", "!", ""])
    if RNG.random() < 0.06:
        text = text.upper() if RNG.random() < 0.3 else text.capitalize()
    return text, spans


def apply_typo(text, spans):
    words = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
    free = [w for w in words if not any(s < w[1] and w[0] < e for s, e, _ in spans)]
    if not free:
        return text, spans
    s, e = RNG.choice(free)
    new = typo(text[s:e])
    delta = len(new) - (e - s)
    text = text[:s] + new + text[e:]
    spans = [(a + delta if a > s else a, b + delta if b > s else b, t)
             for a, b, t in spans]
    return text, spans


def word_tokenize(text):
    return [(m.group(0), m.start(), m.end()) for m in WORD_RE.finditer(text)]


def to_bio(text, spans):
    words = word_tokenize(text)
    tags = ["O"] * len(words)
    for s, e, slot in spans:
        first = True
        for i, (_, ws, we) in enumerate(words):
            if ws >= s and we <= e:
                tags[i] = ("B-" if first else "I-") + slot
                first = False
    return [w for w, _, _ in words], tags


def normalize(text):
    text = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", text).strip()


def emit(rows, split, intent, lang, text, spans):
    text = normalize(text)
    words, tags = to_bio(text, spans)
    if words:
        rows[split].append({"text": text, "intent": intent, "lang": lang,
                            "words": words, "tags": tags})


def build(n_per_intent=1400, oos_ratio=0.10):
    """train and val share templates and synonyms; test gets neither."""
    lex_tr, lex_te = split_lex()
    rows = {"train": [], "val": [], "test": []}
    for intent, langs in TEMPLATES.items():
        for lang in ("en", "el"):
            tpls = langs.get(lang) or []
            if not tpls:
                continue
            n_tpl = len(tpls)
            held = set(RNG.sample(range(n_tpl), max(1, n_tpl // 5))) if n_tpl >= 5 else set()
            keep = [i for i in range(n_tpl) if i not in held]
            n = n_per_intent // 2
            for _ in range(n):
                i = RNG.choice(keep)
                text, spans = render(intent, lang, lex_tr, tpls[i])
                if RNG.random() < 0.12:
                    text, spans = apply_typo(text, spans)
                emit(rows, "train", intent, lang, text, spans)
            for _ in range(max(40, n // 8)):
                i = RNG.choice(keep)
                text, spans = render(intent, lang, lex_tr, tpls[i])
                emit(rows, "val", intent, lang, text, spans)
            for _ in range(max(60, n // 6)):
                i = RNG.choice(sorted(held)) if held else RNG.choice(keep)
                text, spans = render(intent, lang, lex_te, tpls[i])
                emit(rows, "test", intent, lang, text, spans)

    total = len(rows["train"])
    for i in range(int(total * oos_ratio)):
        lang = "en" if i % 2 == 0 else "el"
        base = RNG.choice(OOS[lang])
        if RNG.random() < 0.35:
            base = base + " " + RNG.choice(OOS[lang])
        if RNG.random() < 0.2:
            base = RNG.choice(PREFIX[lang]) + " " + base
        split = "train" if i % 10 < 8 else ("val" if i % 10 == 8 else "test")
        emit(rows, split, "oos", lang, base, [])

    for k in rows:
        RNG.shuffle(rows[k])
    return rows


def main():
    out = Path(__file__).resolve().parent.parent / "data"
    out.mkdir(exist_ok=True)
    rows = build()
    for split, items in rows.items():
        f = out / (split + ".jsonl")
        with f.open("w", encoding="utf-8") as fh:
            for r in items:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{split:5s} {len(items):6d} -> {f}")
    intents = sorted(set(r["intent"] for r in rows["train"]))
    slots = sorted({t[2:] for r in rows["train"] for t in r["tags"] if t != "O"})
    (out / "labels.json").write_text(
        json.dumps({"intents": intents, "slots": slots}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    uniq = len({r["text"] for r in rows["train"]})
    print(f"{len(intents)} intents, {len(slots)} slot types, "
          f"{uniq} unique training utterances")


if __name__ == "__main__":
    main()
