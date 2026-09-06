# Improvement plan, memory neutral

Goal: make BSLM better without growing it. "Memory" here means the model
itself, parameters in RAM and bytes on disk, not the machine. Budget:
5.08M parameters and 20.35 MB fp32 today. Every item below states its memory
delta. Anything marked **+0** changes no weights at all.

Numbers quoted are from `BENCHMARK.md` (regenerate with
`python -m bslm.benchmark`; that file is the scoreboard for this plan).

## Where we are

| measure | now | target |
|---|---|---|
| hand written suite, intent and slots both right | 89.5% | 95% |
| Greek half of the suite | 86.4% | 93% |
| one typo per sentence | 85.3% | 92% |
| Greek typed without accents | 20.5% | 90% |
| Greeklish (latin keyboard) | 4.5% | 85% |
| out of scope wrongly accepted (hand written) | 3 of 16 | 0 of 16 |
| adversarial split (unseen phrasing and verb) | 75.3% | 82% |
| bytes on disk | 20.35 MB | under 11 MB |
| cpu latency | 3.4 ms | under 4 ms |

The ten suite misses split cleanly: four are colloquial assistant chatter
(`what are you able to do`, `άσ' το, δεν πειράζει`), three are slot order or
an unseen value (`from twelve minutes`, `Christmas` tagged as location,
`αύριο` tagged as location), two are Greek out of scope with an in-domain
word, one is a thanks form (`να 'σαι καλά`). None of them need a bigger
model. All of them need data the model has not seen.

## A. Data, +0 parameters

The strongest lever by far. The first synonym layer moved hand written
accuracy from 76% to 100% on the original probe set with zero architecture
change, and that is the pattern to keep repeating.

| id | change | why | gate |
|---|---|---|---|
| A1 | Emit each Greek training row three ways: accented, accent stripped, Greeklish (deterministic transliteration, already in `benchmark.py`). Keep vocab at 4000 and let BPE re-balance. | Phones in Greece type without accents or in latin letters. The two worst rows in the benchmark. | accentless 20% to 90%, Greeklish 4% to 85% |
| A2 | Shuffle slot order inside templates (`remind me at {time} to {task}`, `καιρός {date} {location}`) and add unseen slot values at generation time (holidays, genitive forms like `λεπτών`, made up names). | Three misses are order or value novelty. The slot head has only seen one order per template. | suite slots 96.8% to 99% |
| A3 | Colloquial forms for the six `smalltalk.*` and `assistant.*` intents, both languages, especially Greek ("άσ' το", "τι παίζει", "να 'σαι καλά"). | Four of ten misses. These intents have the fewest templates. | those intents to 100% on the suite |
| A4 | Harder out of scope negatives: Greek gibberish, questions that contain an in-domain word but are not a command ("who is the prime minister" has "who", "what do you think about the weather"). | Three false accepts, all of that shape. Generated OOS is at 100% because it is too easy. | 0 false accepts |
| A5 | Usage flywheel: `assistant.py` and `gui.py` append every utterance under 0.7 confidence to `data/inbox.jsonl`. Review weekly, label, promote to templates. | The only source of phrasings we cannot invent. Costs nothing per row. | adversarial 75% to 82% over time |

Order: A1 and A3 first (biggest gaps, an hour of writing), then A4, A2, A5.
Each is one regeneration and a six minute retrain.

## B. Training recipe, +0 parameters

| id | change | why | gate |
|---|---|---|---|
| B1 | EMA of weights (decay 0.999), evaluate the EMA copy. | Free half a point on small models, smoother slot boundaries. Zero inference cost, the EMA replaces the raw weights. | val joint, suite |
| B2 | R-Drop: two dropout passes per batch, KL between them added to the loss. | Consistency regularisation; the typo row (85%) is a consistency failure. | typo row to 92% |
| B3 | Confusable pair mining: after each epoch, oversample the pairs the model confuses (`timer.query` vs `time.query`, `smalltalk.bye` vs `thanks`, `assistant.cancel` vs `timer.cancel`). | The adversarial confusions are concentrated in a handful of pairs. | adversarial weakest intents |
| B4 | Temperature scaling on the intent logits (one scalar, fit on val) and a margin rule: if top1 minus top2 is under 0.2, ask instead of act. | Confidence is miscalibrated in both directions: gibberish accepted at 0.93, while `will it rain tomorrow`, a verbatim training template, scores 0.38 and is rejected. Short utterances are systematically underconfident. Calibration makes the 0.55 threshold mean something. | false accepts to 0 without losing suite recall; `will it rain tomorrow` accepted |
| B5 | Cross lingual alignment loss: pull the sentence vector of an English row toward a Greek row with the same intent and slot types (cosine, small weight). | English has more synonyms than Greek; alignment lets the Greek side borrow them. Greek trails English by 6 points. | greek suite 86% to 93% |

## C. Architecture, re-allocate instead of add

Only after A and B, and only one at a time, each judged on the benchmark.

| id | change | params | why |
|---|---|---|---|
| C1 | Factorised embeddings: 4000x256 becomes 4000x64 plus 64x256. | **-0.75M** | The embedding table is 20% of the model and mostly memorises rare pieces. ALBERT showed the projection loses almost nothing. Bank the savings, or spend them on C2 and C4. |
| C2 | CRF on top of the slot head. | **+3.2k** | Enforces legal BIO transitions, fixes the split spans (`αύριο` as a second location) without touching the encoder. |
| C3 | Rotary positions instead of the learned 64x256 table. | **-16k** | Generalises to longer inputs; the learned table cannot represent position 65. |
| C4 | Vocab audit: count BPE pieces actually used in training; if the tail is dead, retrain at 3000. | **-0.26M** | Smaller table, faster embedding lookup, no accuracy change if the tail was unused. |
| C5 | Cross layer weight sharing (two unique blocks, three passes each). | **-2.6M** | Halves the model. Costs about a point on this kind of task. Keep in reserve for a device with a hard limit; do not do it for its own sake. |

C1 plus C2 plus C3 plus C4 is about **-1.0M parameters (-20%)** with expected
accuracy flat to slightly up. That is the headroom to spend on one more layer
if A and B ever plateau, and still be under today's size.

## D. Bytes on disk and RAM, no accuracy change

| id | change | disk | latency | note |
|---|---|---|---|---|
| D1 | Save the checkpoint in fp16, upcast on load. | 20.35 to **10.2 MB** | none | Do this now. Compute stays fp32, accuracy is bit for bit the same on this model. |
| D2 | int8 dynamic quantisation of the Linear layers. | **8.3 MB** measured | **worse**, 7.4 ms vs 3.4 ms measured | At 5M parameters the quantise and dequantise overhead beats the matmul saving on x86. Accuracy unchanged. Use only where bytes matter more than milliseconds, or move to static quantisation with calibration. |
| D3 | ONNX export, onnxruntime CPU and WebAssembly. | same as D1 | expected 2 to 3x faster | Removes the torch runtime (hundreds of MB) from the deployment, which is the real memory cost on a phone, not the 20 MB of weights. |
| D4 | Magnitude prune 40% of FFN weights, fine tune two epochs, store sparse. | about 7 MB | none at this size | Only after C, and only if D1 plus D3 is not enough. |

## E. Product layer, +0 parameters

Behaviour that users read as intelligence but that lives in `skills.py`
and `gui.py`, not in the weights.

| id | change | why |
|---|---|---|
| E1 | Context carry over: keep the last intent and slots; if the next utterance is a bare slot ("and the bedroom", "και στην κουζίνα") reuse the intent. | Ellipsis is the most common follow up in a real assistant. |
| E2 | "Did you mean" on a thin margin (B4) instead of a silent wrong action. | A wrong timer cancel is worse than a question. |
| E3 | Split compound commands on conjunctions (" and then ", " και μετά ") and parse each part. | Listed as a limit today; the fix is a string split. |
| E4 | Soul defaults: fill a missing `location` from the user's city, a missing music slot from a stated favourite, greet by name. | Implemented in `gui.py`; extend to the CLI. |

## Order of execution

1. D1 (ten minutes, halves the disk size, nothing else changes).
2. A1 and A3, retrain, benchmark. Expect the two Greek robustness rows to move
   from single digits to the high eighties and the suite to about 93%.
3. A4 and B4 together, benchmark. Expect zero false accepts.
4. B1 and B2, benchmark. Expect the typo row past 90%.
5. A2 and C2, benchmark. Expect suite slots at 99%.
6. B3 and B5 for the adversarial split and the Greek gap.
7. C1, C3, C4 as a block, benchmark, keep only if flat or better.
8. E1 to E3 in `skills.py`, independent of everything above.
9. A5 runs from step 2 onwards and never stops.

Every step is gated on `BENCHMARK.md`: a change that does not move its row
is reverted, whatever the theory said.

## What not to do

- Do not add layers or width to chase the misses. Every miss above is a
  data gap; a bigger model memorises the same gap more confidently.
- Do not bolt on a pretrained encoder. The smallest useful one (MiniLM L6)
  is 22M parameters and 90 MB, four times this whole project, and it
  would make the from scratch claim false.
- Do not raise the vocabulary past 4000 to "fix" Greeklish. A1 handles it
  inside the current table; measure before spending 256k parameters per
  thousand tokens.
- Do not tune on the hand written suite. It is the test. Write new
  sentences for development, or the numbers stop meaning anything.
