"""Hand written probes: phrasings that appear nowhere in the training templates.

This is the honest test. The held out split still reuses the generator's
vocabulary; these sentences were written by hand to break it.
"""
import sys

from .infer import Parser

PROBES = [
    # (utterance, expected intent)
    ("could you start counting down from twelve minutes", "timer.set"),
    ("buzz me in 3 min", "timer.set"),
    ("kill that countdown", "timer.cancel"),
    ("get me up at 6:45 tomorrow", "alarm.set"),
    ("scrap the alarm", "alarm.cancel"),
    ("nudge me to pay the electricity bill on Friday", "reminder.create"),
    ("what have I got to do this week", "reminder.list"),
    ("stick a dentist appointment in for Tuesday at 9:00", "calendar.create"),
    ("am I busy on Thursday", "calendar.query"),
    ("is it going to pour down in Patras tomorrow", "weather.query"),
    ("chuck on some blues", "music.play"),
    ("shut the music up", "music.control"),
    ("crank the volume to 80", "volume.set"),
    ("kill the lights in the garage", "light.control"),
    ("switch the boiler on", "device.control"),
    ("stick yogurt on the grocery list", "list.add"),
    ("read out the shopping list", "list.read"),
    ("jot down that the spare key is under the pot", "note.create"),
    ("what have I written down", "note.read"),
    ("drop Maria a line saying I am running late", "message.send"),
    ("get my landlord on the phone", "call.make"),
    ("how do I drive to Volos", "navigation.route"),
    ("look up cheap hotels in Rome", "search.web"),
    ("what is thank you very much in Japanese", "translate"),
    ("work out 348 divided by 12", "math.calculate"),
    ("how many miles is 42 kilometers", "unit.convert"),
    ("got the time", "time.query"),
    ("anything new on the economy", "news.query"),
    ("morning", "smalltalk.greet"),
    ("cheers mate", "smalltalk.thanks"),
    ("write me a sonnet about the moon", "oos"),
    ("explain how a nuclear reactor works", "oos"),
    ("qwerty zxcv", "oos"),
    # greek, hand written
    ("βάλε μου χρονόμετρο δεκαπέντε λεπτών", "timer.set"),
    ("σταμάτα το να μετράει", "timer.cancel"),
    ("σήκωσέ με στις 7 το πρωί", "alarm.set"),
    ("θύμισέ μου να ποτίσω τα φυτά την Τετάρτη", "reminder.create"),
    ("τι έχω να κάνω αύριο", "reminder.list"),
    ("κλείσε ραντεβού στον γιατρό την Πέμπτη στις 10:30", "calendar.create"),
    ("θα βρέξει στη Λάρισα το σαββατοκύριακο", "weather.query"),
    ("βάλε λίγα ρεμπέτικα", "music.play"),
    ("κόψε τη μουσική", "music.control"),
    ("δυνάμωσε λίγο", "volume.set"),
    ("σβήσε τα φώτα στο μπάνιο", "light.control"),
    ("άναψε τον θερμοσίφωνα", "device.control"),
    ("πρόσθεσε γιαούρτι στη λίστα με τα ψώνια", "list.add"),
    ("σημείωσε ότι το τιμολόγιο είναι 4471", "note.create"),
    ("στείλε μήνυμα στον Νίκο ότι θα αργήσω", "message.send"),
    ("πάρε τον λογιστή", "call.make"),
    ("πώς πάω στο Ηράκλειο", "navigation.route"),
    ("πόσο κάνει 144 δια 12", "math.calculate"),
    ("πόσα κιλά είναι 30 λίβρες", "unit.convert"),
    ("τι ώρα είναι τώρα", "time.query"),
    ("γράψε μου ένα τραγούδι για τη θάλασσα", "oos"),
]


def main():
    p = Parser()
    ok = 0
    bad = []
    for text, want in PROBES:
        r = p.parse(text)
        good = r["intent"] == want
        ok += good
        mark = "ok " if good else "MISS"
        slots = " ".join(f"{k}={v}" for k, v in r["slots"].items())
        print(f"{mark} {r['intent']:22s} {r['confidence']:.2f}  {text}")
        if slots:
            print(f"     slots: {slots}")
        if not good:
            bad.append((text, want, r["intent"], r["confidence"], r["top3"]))
    print(f"\n{ok}/{len(PROBES)} = {ok/len(PROBES):.1%} on unseen hand written phrasings")
    if bad:
        print("\nmisses:")
        for t, w, g, c, top in bad:
            print(f"  '{t}'  want {w}  got {g} ({c:.2f})  top3 {top}")
    return 0 if ok == len(PROBES) else 1


if __name__ == "__main__":
    sys.exit(main())
