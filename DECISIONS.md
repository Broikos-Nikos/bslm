# The decisions, and why

A short reading of how BSLM got here. The full dated log is `PROCESS.md`; this
is the distilled version, the forks that mattered and the reason each way was
taken.

## From scratch, no pretrained weights
The goal was to understand a small language model by building one, not by
wrapping an existing one. So every weight starts from random init and every
byte of training data is generated or fetched by this repo. A pretrained 0.6B
model was tried once as a yardstick and rejected after it hallucinated a fact
with full confidence; the point was a model that knows how to check, not one
that memorised.

## Two models, not one
The classic assistant core (intent plus slots) is tiny and solvable at 5M
parameters, so that is the **router**. Everything that needs judgement, looking
things up, retrying, holding a short memory, is a separate **72M loop model**.
Keeping them apart means the cheap path stays cheap and the expensive path only
runs when it is needed.

## The 80 MB phone cap set the size
The model had to fit a phone with room to spare, so the ceiling was fixed at
80 MB before training began. That is 56M to 72M parameters at Q8_0. 72M won.
Later a size sweep confirmed facts plateau at the same place for 56M and 72M,
so the cap did not cost accuracy: the fact ceiling is the task, not the size.

## Quality over quantisation
Held out perplexity showed Q8_0 is lossless within noise (17.32 vs 17.30 for
f16) while Q4_0 costs about 8.5%. The owner does not accept quality loss, so
Q8_0 ships on every device, and the phone gets its speed from prefix caching
and Arm kernels, not a smaller number format.

## Teach the loop from real trajectories
The intelligence is trained, not scripted. The generator plays out real tool
calls (Wikipedia search and pages, Open-Meteo, YouTube, the local skills) where
the answer is known in advance, including real failures with the correct
recovery written out. The model learns the loop by seeing thousands of them.
A coded check backs the one step the model is weakest at, delivery, so a fact
it cannot confirm from the page is not delivered as fact.

## Honest measurement, a frozen benchmark
Every family is scored against an 80% bar on a held out set frozen since round
six, and the benchmark reports the model alone and with the delivery check.
When rounds started fighting sample noise rather than real signal, that was
called out and the round loop stopped, rather than chasing a moving number.

## Local compute, rent only at the edge
Training is local on one RTX 4060 by default. A 10B token run was rented once
on an H100; afterwards the policy became local first, rent only if a run would
take more than three days and is needed sooner, cheapest tier, never an H100
for these sizes again. Per token the local card is about five times cheaper.

## Shipping v1
Round seven is v1: 91% on the agent benchmark, every task family passing except
open fact lookup at about 70% (the measured ceiling), delivered with the source
quote. Facts do not clear 80% because that needs a bigger model, which the
phone cap rules out, so v1 ships as it is and the size and recipe levers are
recorded as exhausted. Wrappers (the PC assistant, the Android shell, the smart
home) came after the model, never before it.
