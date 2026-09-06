# Building the language model yourself

No pretrained weights anywhere in the pipeline, and no large model generating
the training data either. Everything the model learns comes from public text
and from programs. This document is the way to do that optimally for the goal
you described: a phone sized model that does not know the answer but knows
how to check. English only.

## The one decision that makes it feasible

The model must learn **language** (read a web page, understand a request,
follow a tool protocol, write a short answer) and **procedure** (search, open
two or three results, compare, pick). It must not learn **facts**. Facts come
from tools at run time.

That decision is what separates a possible project from an impossible one.
Knowledge is what costs billions of parameters and trillions of tokens.
Language competence inside a bounded domain is cheap: TinyStories showed 10M
parameter models writing coherent English when the distribution is narrow,
and GPT-2 small (124M) reads and summarises web text after 10B tokens.

The target is therefore a **56M to 72M parameter decoder** (the 80 MB cap),
trained on **5 to 10B tokens seen**, then taught the assistant job on synthetic
trajectories whose tool results are real. At Q8 that is 59 to 77 MB on the phone.

## What the numbers say

Training cost is 6 x parameters x tokens FLOPs. The RTX 4060 sustains about
15 TFLOPS in practice on a small model (bf16, torch.compile, fused attention,
about 30% utilisation).

**Hard cap, decided 2026-09-04: at most 80 MB on the phone.** At Q8_0
(about 1.06 bytes per parameter, the quantisation that costs no quality at
this size) that is 56M to 72M parameters. Only these two sizes exist in
`pretrain/model.py`; nothing bigger will be trained. Quality comes from
tokens, data and recipe, not from parameters.

| model | Q8 size | tokens seen | RTX 4060, measured or projected |
|---|---|---|---|
| 56M (d640, 10 layers) | 59 MB | 1.2B | 6.5 h measured (52k tokens per second) |
| 56M | 59 MB | 5B | about 27 h |
| 72M (d768, 10 layers) | 77 MB | 5B | about 36 h |
| 72M | 77 MB | 10B | about 3 days |

Everything runs on the 4060 by default. One 10B token run was bought on a
rented H100 (2026-09-06, 3.49 USD/h, 16 times the 4060) and judged not
worth the money: per token the local card is about five times cheaper than
any rental, renting only buys wall clock time. Rule since then: rent only
when a run would take more than three days locally and the result is
needed sooner, and then the cheapest tier that fits (a Community Cloud
4090 at about 0.5 USD/h is the only tier with a defensible tokens per
dollar). The same code runs in both places (`pretrain/remote/`). The recipe
below is what makes one consumer card enough:

- **The modern recipe is not optional.** Muon for the hidden matrices, bf16,
  fused attention, torch.compile, sequence packing. The modded-nanogpt line
  reaches GPT-2 quality with three to four times fewer tokens than 2019
  did, and on one card that is the difference between a week and a month.
- **Tokens seen, not unique tokens.** A curated 3 to 4B token corpus read
  three times is nearly as good as 10B unique tokens (data constrained
  scaling, Muennighoff 2023). Curation is cheap; unique tokens are not.
- **Interruptible by design.** Warmup, stable, decay schedule and hourly
  checkpoints, so a run survives reboots and shares the card with whatever
  else needs it; the decay phase is run at the end, on whatever checkpoint
  is best.
- **8 GB is enough.** A 110M model with gradient accumulation to 0.5M tokens
  per step fits in about 5 GB; 280M fits with activation checkpointing.
- **Size is capped, so tokens do the work.** 56M overnight proves the
  pipeline. The remaining levers are tokens seen (multi epoch on curated
  data), the data mix, the recipe, and the step from 56M to 72M, which is
  the last 20 MB of the budget.

## Stage 1. Corpus, 10 to 30B tokens, all public

| source | share | why |
|---|---|---|
| FineWeb-Edu (English web, quality filtered) | 60% | general language, reading comprehension |
| Wikipedia, English | 10% | the shape of the pages it will open when it searches |
| OpenSubtitles, English | 10% | dialogue, short turns, colloquial forms |
| raw fetched pages and search result lists, captured by our own crawler | 12% | tool output as the model will actually see it: titles, snippets, boilerplate, navigation noise |
| a little code (The Stack, permissive) | 8% | structure and JSON; measurably helps tool calling |

None of this involves a language model. A crawler, a parser and a
deduplicator are programs.

## Stage 2. Tokenizer

Train a BPE from scratch on the mix. English only keeps the table small:
test 16k against 32k, because at d768 the smaller vocabulary saves 12M
parameters (tied embeddings are a fifth of a 110M model) and costs almost
nothing in tokens per word for English.

## Stage 3. Architecture

Llama shaped, so that llama.cpp runs it on the phone with zero extra work:
RMSNorm, rotary positions, SwiGLU, grouped query attention, tied input and
output embeddings, context 2048. Written by hand as `bslm/model.py` was.
Two sizes, 56M and 72M, both under the 80 MB cap at Q8.

## Stage 4. Pretraining recipe

- bf16, torch.compile, fused attention, gradient clipping 1.0
- Muon for the hidden matrices, AdamW for embeddings and norms
- warmup, stable, decay schedule, so a run can be extended without restarting
- 0.5M tokens per step, sequence 1024, checkpoints hourly
- **mid training** in the last 15%: raise the share of tool output text
  (fetched pages, result lists, calendar dumps, weather JSON, YouTube
  result lists) so reading tool output is native, not learned later

Loss on held out text is the only metric until stage 6.

## Stage 5. Teaching the job, with no language model in the loop

This is where the from scratch data engine you already have pays off. The
template engine becomes a **trajectory generator**: multi turn dialogues of
user request, tool call, tool result, final answer. The trick that makes it
work without an LLM is that **the tool results are real and the answers are
known before the question is asked**:

- **Search and read.** Take 50k facts from Wikidata (head of government of
  X, capital of Y, birth year of Z, height of W). Render each as questions
  by template, many phrasings each. Run the real search, capture the real
  result list and the real top pages. The correct answer is in Wikidata, so
  the target reply is written by a rule. The model learns to find a known
  string inside noisy real pages, which is exactly the job.
- **Play a song.** For 20k real song titles, capture the real YouTube
  result list. The correct video is the one whose title matches the song
  and artist (a rule). The model learns to pick it from the list; a
  program then navigates to it and it plays.
- **Agenda.** Generate calendars, ask about them, answer by rule.
- **The umbrella.** Generate calendars with events that have a location,
  fetch real hourly forecasts from Open-Meteo for those places and times,
  and render the advice by rule: rain probability above 40% one or two
  hours before an outdoor event, plus the soul fact that the user rides a
  scooter, produces "take the car" or "take an umbrella". The model
  learns to say it; a scheduler decides when to ask it.
- **Compound requests.** Two templates joined by a conjunction, two tool
  calls expected.
- **Failure and recovery, deliberately.** Trajectories are generated with
  real failures baked in: a first query that returns nothing (rendered from
  a bad template on purpose), a page that does not contain the fact, a tool
  that returns an error string, an ambiguous request that needs a question
  back. The generator knows the truth, so it also knows the correct
  judgement ("the page is about the 2019 election, not the current one")
  and the correct next move. Distribution: 40% succeed first time, 35% need
  one retry, 20% need two, 5% cannot be done and end with an honest "here
  is what I found and what I could not". The model learns the loop by
  seeing the loop, thousands of times, with the judgement written out.

Around 100k trajectories, fine tuned for one epoch. This is supervised
learning on data whose labels come from databases and rules, which is the
same thing the 5M model did, one level up.

## Stage 6. The loop: try, judge, retry, deliver

This is the intelligence you asked for, and it is trained, not scripted. The
runtime provides only the turn taking: the model acts, the environment
answers, the model speaks again. Every decision inside is the model's.

```
[plan]     what I will do first and why
[act]      search("...") | open(url) | read | compute | act(device) | ask(user)
[observe]  the raw result, exactly as the environment returned it
[judge]    success | partial, and what is missing | failure, and why
[act]      the next attempt, changed because of the judgement
           ... up to three attempts per subgoal ...
[verify]   does what I have answer the request as it was asked
[deliver]  the answer, or what was achieved and what was not
```

The actions are the model's hands, not wrappers: search, open, read,
compute, act, remember, ask. What stands behind an action on a given phone
(which browser, which calendar, which home system) is the wrapper, and it
comes after. The intelligence is choosing the action, reading what came
back, saying what went wrong, and knowing when it is done.

The judge step is what separates this from a chain of tool calls: the model
writes down why the attempt failed before choosing the next move, and it is
trained on trajectories where that sentence is correct. Three attempts per
subgoal, then deliver honestly.

Where the model measurably fails at a step (stage 7 says which), a coded
procedure takes that step over. Code is the fallback, not the design.

## Stage 7. Measure, then choose the size

The scenario list is `SCENARIOS.md` (generated from `bslm/scenarios.py`,
Siri's capabilities with a status per item); every R and A row becomes
benchmark cases. Extend `bslm/benchmark.py` with an agent section: search QA accuracy on
1,000 held out Wikidata questions, song pick accuracy on 500 held out titles,
compound command completion and the umbrella scenario. Train 56M and 72M on the same data; ship the one that clears the bar,
and if both do, the smaller one. That is what "optimally" means in practice: the
size is an output of the measurement, not an input.

The loop gets its own metrics, because a model can score well on single
step tasks and still be useless at recovery:

| metric | what it catches | bar |
|---|---|---|
| recovery rate | success after a failed first attempt | 80% |
| false delivery | says done when the result does not answer the request | under 3% |
| wasted steps | repeats a failed action unchanged | under 10% |
| honest give up | when it cannot be done, says so instead of inventing | 95% |

## Stage 8. The phone

Export to GGUF (the architecture is Llama shaped, so the standard converter
works), quantise to Q8 for the 56M and 110M models or Q4_K_M for the 280M,
run under llama.cpp on device. The runtime is already wired: `reasoner.py`
talks to llama-server and does not care whose weights are inside.

## What to expect, honestly

- **110M, 10B tokens, two to three days on the 4060:** follows the tool protocol, reads a result list and
  extracts a named answer, plays the right song, reads the calendar. Brittle
  when the answer is not a literal string on the page.
- **280M, 10B tokens, a week on the 4060:** handles every scenario you listed with the
  procedures above, including the advice phrasing.
  Still cannot write an essay or reason about a situation it never saw in
  the trajectories. That is not the job.
- **Neither** will ever match a 3B pretrained model at open ended chat.
  The goal was never that.

## First step

Build the pretraining pipeline (`pretrain/`: corpus download and shard,
tokenizer training, the Llama shaped model, the training loop) and run the
56M model on 1 to 2B tokens overnight on the 4060. That proves the whole
chain, produces the first samples of your own model reading English, and gives the throughput number that makes the rest of the plan
exact instead of estimated.
