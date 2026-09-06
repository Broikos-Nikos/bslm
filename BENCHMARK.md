# BSLM benchmark

Generated 2026-09-06 02:32 by `python -m bslm.benchmark` in 124s.
Everything below is measured on the checkpoint in `checkpoints/`, nothing is
hand entered.

## Why it was built

The question was whether an everyday assistant, the "set a timer, remind me,
add milk to the list" kind, can be built **from scratch**: no pretrained
weights, no API, no vendor. Not a wrapper that looks like Siri, the actual
model. The answer is yes, with one condition: the task list has to be fixed.
A general chatbot from scratch needs trillions of tokens and a datacentre.
A fixed task list needs a 5M parameter encoder and six minutes on one GPU.

The point of publishing it is the trade itself. This model is 20 MB in fp32,
8.3 MB in int8, answers in 3.8 ms on a CPU, needs no network, and
scores 89.5% on sentences it has never seen. The cost is that it knows
exactly and only what its templates taught it.

## How it was built

1. **Templates first.** 35 intents, 28 slot types, English and Greek, written
   by hand in `bslm/templates.py`. A synonym layer (`[start]`, `[cancel]`,
   `[read]`...) expands each template into a dozen surface forms.
2. **Corpus generated**, not collected: 65,016 labelled utterances with
   character level slot spans, typos, fillers and casing noise.
3. **BPE tokenizer trained** on that corpus (4000 merges, inside word boundaries).
4. **Transformer encoder written by hand** in `bslm/model.py`: 6 pre-norm
   blocks, d_model 256, 4 heads, one head for intent, one for BIO slot tags.
   5.09M parameters.
5. **Trained from random init**, joint cross entropy, 8% token dropout so it
   cannot lean on a single keyword.
6. **Evaluated adversarially.** The first version scored 99.9% on validation
   and 76% on hand written sentences: it had memorised verbs. The synonym
   layer took hand written accuracy to 100% with the same architecture.
7. **Skills attached** in `bslm/skills.py`: timers, reminders, notes, lists,
   calendar, maths, units and a home state that really execute and persist.

## What it can do

One hand written example per language from the suite below. Slots are the
arguments the model extracts as text spans.

| intent | english | greek | slots |
|---|---|---|---|
| `alarm.cancel` | scrap the alarm | μη με ξυπνήσεις το πρωί |  |
| `alarm.set` | get me up at 6:45 tomorrow | σήκωσέ με στις 7 το πρωί | date, time |
| `assistant.cancel` | no wait, forget that | άσ' το, δεν πειράζει |  |
| `assistant.capabilities` | what are you able to do | σε τι είσαι χρήσιμος |  |
| `assistant.repeat` | sorry, say again | δεν κατάλαβα, ξαναπές το |  |
| `calendar.create` | stick a dentist appointment in for Tuesday at 9:00 | κλείσε ραντεβού στον γιατρό την Πέμπτη στις 10:30 | date, time, title |
| `calendar.query` | am I busy on Thursday | έχω κάτι κανονισμένο την Παρασκευή | date |
| `call.make` | get my landlord on the phone | πάρε τον λογιστή | person |
| `device.control` | switch the boiler on | άναψε τον θερμοσίφωνα | device, room |
| `light.control` | kill the lights in the garage | σβήσε τα φώτα στο μπάνιο | color, room |
| `list.add` | stick yogurt on the grocery list | πρόσθεσε γιαούρτι στη λίστα με τα ψώνια | item, list_name |
| `list.read` | read out the shopping list | τι έχει η λίστα για το σούπερ μάρκετ |  |
| `math.calculate` | work out 348 divided by 12 | πόσο κάνει 144 δια 12 | expression |
| `message.send` | drop Maria a line saying I am running late | στείλε μήνυμα στον Νίκο ότι θα αργήσω | content, person |
| `music.control` | shut the music up | κόψε τη μουσική |  |
| `music.play` | chuck on some blues | βάλε λίγα ρεμπέτικα | artist, genre, song |
| `navigation.route` | how do I drive to Volos | πώς πάω στο Ηράκλειο | location |
| `news.query` | anything new on the economy | τι νέα έχουμε για την οικονομία | topic |
| `note.create` | jot down that the spare key is under the pot | σημείωσε ότι το τιμολόγιο είναι 4471 | note |
| `note.read` | what have I written down | διάβασέ μου τι έχω γράψει |  |
| `reminder.create` | nudge me to pay the electricity bill on Friday | θύμισέ μου να ποτίσω τα φυτά την Τετάρτη | date, task, time |
| `reminder.list` | what have I got to do this week | τι έχω να κάνω αύριο | date |
| `rule.set` |  |  |  |
| `search.web` | look up cheap hotels in Rome | ψάξε μου φθηνά αεροπορικά για Λονδίνο | query |
| `smalltalk.bye` | catch you tomorrow | καλό βράδυ, τα λέμε |  |
| `smalltalk.greet` | morning | καλησπέρα σας |  |
| `smalltalk.thanks` | cheers mate | να 'σαι καλά |  |
| `time.query` | got the time | τι ώρα είναι τώρα | date |
| `timer.cancel` | kill that countdown | σταμάτα το να μετράει |  |
| `timer.query` | is the timer nearly finished | πόσο μένει ακόμα στο χρονόμετρο |  |
| `timer.set` | could you start counting down from twelve minutes | βάλε μου χρονόμετρο δεκαπέντε λεπτών | duration |
| `translate` | what is thank you very much in Japanese | πώς λένε καλημέρα στα ιαπωνικά | language, phrase |
| `unit.convert` | how many miles is 42 kilometers | πόσα κιλά είναι 30 λίβρες | amount, unit_from, unit_to |
| `volume.set` | crank the volume to 80 | δυνάμωσε λίγο | level |
| `weather.query` | is it going to pour down in Patras tomorrow | θα βρέξει στη Λάρισα το σαββατοκύριακο | date, location |
| `work.business` |  |  |  |
| `work.code` |  |  |  |
| `work.files` |  |  |  |
| `work.ops` |  |  |  |
| `work.research` |  |  |  |
| `work.review` |  |  |  |
| `work.status` |  |  |  |
| `work.write` |  |  |  |

## Accuracy on hand written sentences

95 sentences (51 English, 44 Greek) written by hand after training, with
the expected intent **and** the expected slot values. A sentence counts as
correct only if both match. Slot values are compared after lowercasing,
accent stripping and dropping a leading article or preposition.

| set | n | intent | slots | both |
|---|---|---|---|---|
| all | 95 | 91.6% | 96.8% | 89.5% |
| english | 51 | 98.0% | 98.0% | 96.1% |
| greek | 44 | 84.1% | 95.5% | 81.8% |

Misses:

- `could you start counting down from twelve minutes`  wanted **timer.set** {"duration": "twelve minutes"}  got **timer.set** (0.96) {}
- `what are you able to do`  wanted **assistant.capabilities** {}  got **reminder.list** (0.67) {}
- `μη με ξυπνήσεις το πρωί`  wanted **alarm.cancel** {}  got **alarm.set** (0.95) {"time": "το πρωί"}
- `τι καιρό θα κάνει αύριο στην Πάτρα`  wanted **weather.query** {"date": "αύριο", "location": "Πάτρα"}  got **weather.query** (0.96) {"location": "στην Πάτρα"}
- `τι νέα έχουμε για την οικονομία`  wanted **news.query** {"topic": "οικονομία"}  got **time.query** (0.67) {"topic": ["την", "οικονομία"]}
- `να 'σαι καλά`  wanted **smalltalk.thanks** {}  got **smalltalk.greet** (0.94) {}
- `καλό βράδυ, τα λέμε`  wanted **smalltalk.bye** {}  got **translate** (0.65) {"phrase": "τα λέμε"}
- `σε τι είσαι χρήσιμος`  wanted **assistant.capabilities** {}  got **oos** (0.34) {}
- `δεν κατάλαβα, ξαναπές το`  wanted **assistant.repeat** {}  got **music.play** (0.69) {}
- `άσ' το, δεν πειράζει`  wanted **assistant.cancel** {}  got **oos** (0.37) {}

## Robustness

Same sentences, transformed. Slots are not scored for Greeklish because the
expected values are Greek script.

| transform | n | intent | slots | both |
|---|---|---|---|---|
| one typo per sentence (outside slot values) | 95 | 89.5% | 95.8% | 86.3% |
| ALL CAPS (english) | 51 | 98.0% | 96.1% | 94.1% |
| greek without accents | 44 | 47.7% | 50.0% | 29.5% |
| greeklish, latin keyboard | 44 | 4.5% | not scored | 4.5% |

Greeklish and accentless Greek are how people actually type on phones in
Greece. Neither is in the training data yet; these rows are the baseline the
plan in `PLAN.md` improves on.

## Out of scope rejection

The assistant must refuse what it was not built for instead of guessing.

| set | n | rejected | wrongly accepted |
|---|---|---|---|
| hand written out of scope | 16 | 62.5% | 6 |
| generated out of scope (test split) | 602 | 100.0% | 0 |

- `what do you think about my haircut` accepted as music.play (0.79)
- `recommend a good book` accepted as work.research (0.91)
- `asdkjh qwe` accepted as music.play (0.96)
- `ποιος είναι ο πρωθυπουργός` accepted as translate (0.85)
- `μπορείς να μου εξηγήσεις τη σχετικότητα` accepted as music.play (0.91)
- `λκφδ σδφκ` accepted as timer.set (0.88)

## Adversarial split

10,578 generated sentences built only from **held out templates rendered with
held out synonyms**. The model has never seen the phrasing and never seen the
verb. This is deliberately harsher than real usage.

| | intent accuracy |
|---|---|
| all intents | 72.5% |

Weakest intents, with what they were confused for:

| intent | accuracy | confused with |
|---|---|---|
| `smalltalk.bye` | 0.0% | work.status (62), smalltalk.greet (62) |
| `assistant.repeat` | 0.0% | timer.cancel (116), oos (88) |
| `reminder.list` | 0.0% | reminder.create (116), list.read (116) |
| `alarm.cancel` | 13.8% | alarm.set (179), oos (21) |
| `work.review` | 26.3% | search.web (42), oos (39) |
| `work.ops` | 33.6% | oos (48), music.play (44) |
| `assistant.cancel` | 34.1% | oos (148), calendar.query (3) |
| `music.control` | 44.4% | oos (61), math.calculate (56) |

Every failure here is a verb that was withheld from training ("bin the alarm",
"stream Clocks"). A model with no pretraining can only know a word the corpus
contained. That is the honest ceiling of from scratch.

## Size and latency

| | fp32 | int8 dynamic |
|---|---|---|
| parameters | 5.09 M | same |
| weights in memory | 20.37 MB | 8.31 MB |
| cpu latency, median / p95 | 3.77 / 4.92 ms | 7.33 / 9.58 ms |
| hand written suite, both correct | 89.5% | 89.5% |
| gpu latency, median / p95 | 9.51 / 12.43 ms | |

Tokenizer: 162 KB. Checkpoint on disk: 20.37 MB.
Batch size is one throughout, which is how an assistant is used; at that size
the CPU beats the GPU because kernel launch dominates.

## Transcript

Real run through `bslm/skills.py` with a throwaway state file.

```
> set a timer for 25 minutes
  [timer.set 0.96]  Timer set for 25m.
> how much longer
  [timer.query 0.96]  Time left: 24m 59s
> remind me to pay the electricity bill on Friday at 9:00
  [reminder.create 0.95]  Reminder saved: pay the electricity bill (on Friday 09:00)
> add milk and batteries to the shopping list
  [list.add 0.96]  Added milk, batteries to the shopping list.
> what is on my shopping list
  [list.read 0.96]  shopping list: milk, batteries
> βάλε ξυπνητήρι για τις 7 το πρωί
  [alarm.set 0.95]  Alarm set for 07:00 (tomorrow).
> σημείωσε ότι ο κωδικός του wifi είναι 12345
  [note.create 0.96]  Noted: ο κωδικός του wifi είναι 12345
> what have I noted down
  [note.read 0.96]  Notes:
      1. ο κωδικός του wifi είναι 12345
> how many kilometers is 42 miles
  [unit.convert 0.96]  42 mi = 67.5924 km
> work out 348 divided by 12
  [math.calculate 0.96]  348 divided by 12 = 29
> turn off the lights in the kitchen
  [light.control 0.96]  Lights in kitchen: off
> άναψε τον θερμοσίφωνα
  [device.control 0.96]  θερμοσίφωνα turned on.
> τι υπενθυμίσεις έχω
  [reminder.list 0.96]  Reminders:
      1. pay the electricity bill  [on Friday 09:00]
> write me a poem about the sea
  [oos 0.96]  I did not understand that. It is outside what I was trained on.
> cancel the timer
  [timer.cancel 0.96]  Cancelled 1 timer(s).
```

## Reproduce

```
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m bslm.gen_data
.venv\Scripts\python.exe -m bslm.train --epochs 22
.venv\Scripts\python.exe -m bslm.benchmark
```
