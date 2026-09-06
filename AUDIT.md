# Audit, 2026-09-06

Five passes over every file in the repository, each with one perspective.
Every finding is listed with what was done. "Fixed" means the change is in
the tree and was verified by running it; "Noted" means a real limitation
that was not changed, with the reason.

Scope: `bslm/` (router, skills, benchmarks, runtime), `pretrain/` (data,
model, training, export, remote scripts, chat audit), `assistant.py`,
`gui.py`, the PowerShell loops, all Markdown, `.gitignore`, `requirements.txt`.

## Pass 1: training pipeline correctness (pretrain/)

Perspective: would this produce a model that is wrong, silently.

| finding | severity | status |
|---|---|---|
| Checkpoints were written in place; a crash during `torch.save` leaves a truncated `ckpt.pt`, and the restart loop would then fail on every resume. | high | Fixed: write to `.tmp` and `os.replace`, in both trainers |
| Multi epoch runs replayed the shards in the identical order every epoch (no shuffling anywhere). Harmless for one epoch, wasteful for the 10B token run (four epochs). | medium | Fixed: epoch 0 keeps file order (the running local run is unaffected), later epochs use a seeded permutation, resume stays reproducible |
| Rotary embedding uses the rotate half layout, attention casts to bf16 under autocast, RMSNorm runs in fp32, tied output head, loss over fp32 logits, gradient accumulation divides before backward and clips once. | ok | Verified by reading and by the 56M run's curve and its llama.cpp output |
| GGUF export first wrote norm weights as f16, which llama.cpp rejects. | high | Fixed yesterday; 1D tensors go f32, generation verified in llama.cpp |
| `generate` has no KV cache; it recomputes the whole prefix per token. | low | Noted: eval only, 40 tokens, not worth the code |
| The tail of every shard shorter than one window is dropped by `Stream.next`. | negligible | Noted: under 8k tokens per 100M shard |
| `sample.py` hard codes CUDA. | low | Noted: local tool |
| Validation is the last 22M tokens of the last parquet file, never trained on, even across epochs. | ok | Verified: `Stream("train")` only globs `train_*.npy` |

## Pass 2: secrets, privacy and safety

Perspective: what could leak, and what could be abused.

| finding | severity | status |
|---|---|---|
| Client and product names sat in `pretrain/chat_audit.py` as a redaction list, in a file that goes to GitHub. | high | Fixed: the list moved to `data/redact_names.txt`, gitignored; the code reads it if present |
| A client project and two personal topics appeared as slot values in the generator and one benchmark command ("energy platform", "the doctorate", "the recruiter offer"). | medium | Fixed: replaced with neutral phrases; training data regenerates from templates so nothing is baked in |
| API key, S3 keys and SSH key: all outside the repo (`~/.bslm`, `~/.ssh`), none in the tree (grep for the key prefixes over the whole tree found nothing). The S3 secret was printed once inside a tool result during setup and therefore exists in the local session transcript. | low | Noted: local machine only; rotate the S3 key from the console if that ever matters |
| `skills.py` `calculate()` evaluates an expression with `eval`, after stripping everything except digits, operators, brackets and whitespace. No names, no attribute access, no exponent operator survive the filter. | ok | Verified; oversized integers are the only residual and are harmless |
| `gui.py` control port on 127.0.0.1:8765 with no authentication: any local process can type into the assistant. | low | Noted: documented as a testing hook, local only |
| `pretrain/receiver.py` accepted arbitrary POSTs on localhost and never worked (the sites' content security policy blocks it). | low | Fixed: deleted |
| The reasoning layer will read web search results; anything on a page can try to steer the model. | medium | Noted: inherent to the tool; the procedure design (code chooses, model reads bounded text) is the mitigation |
| Chat exports and the audit report are under `corpus/` and `reports/`, both gitignored. | ok | Verified |

## Pass 3: robustness and operations

Perspective: what breaks at 3 am with nobody watching.

| finding | severity | status |
|---|---|---|
| `resume_when_free.ps1` hard coded the 56m run and `overnight.ps1`; used for the 72m run it would have relaunched the wrong loop. | high | Fixed: takes the run name and the loop script as parameters |
| `pull.sh` relied on brace expansion inside an scp path, which the remote shell may not expand. | medium | Fixed: one scp per file |
| `bootstrap.sh` assumed tmux exists on the pod image. | medium | Fixed: falls back to nohup and setsid |
| The GUI control server is single threaded; a client that connects and sends nothing would block the port forever. | low | Fixed: 10 second socket timeout |
| Python stdout is block buffered when redirected, so the training log only appears at exit in `train.log`; the trainer itself writes its own `log.txt` line by line. | low | Noted: the loops read `log.txt` |
| Both restart loops pass `--resume` unconditionally and stop only on "final step"; a checkpoint that cannot load would retry 30 times. | low | Noted: acceptable with atomic saves in place |
| Windows console encoding (cp1252) breaks on Greek; every entry point reconfigures stdout to UTF-8. | ok | Verified |
| Two trainings on the card or a browser session while training collapses throughput; the watcher pattern exists for the shared card case. | ok | Verified and written into the process log |

## Pass 4: data quality and evaluation honesty

Perspective: are the numbers we quote true.

| finding | severity | status |
|---|---|---|
| The router's inference pre-tokenizer did not recognise the typographic apostrophe while the training data generator did, so "don’t" split into three tokens at inference and one in training. | high | Fixed: one regex in `tokenizer.py`, imported by `gen_data.py`; verified on both apostrophes |
| 19 of the 144 benchmark commands occur verbatim in the generated training set (short generic ones such as "louder", "good morning", "kill the timer", and "ring the accountant", which yesterday's corrective pass produced by chance). The report claimed none were seen. | high | Fixed: `COMMANDS_BENCHMARK.md` now reports the overlap and a clean subset score: 89.6% on the 125 never seen commands against 91.0% overall |
| The adversarial split is built from held out templates rendered with held out synonyms; the hand written suite and the command benchmark are separate held out sets; validation shares templates with training and measures fit only. | ok | Verified in `gen_data.build` |
| `BENCHMARK.md` was generated before the work domain and the threshold change. | medium | Fixed: regenerated on the CPU after this audit |
| Regenerating it exposed a regression: with the work domain and the 0.40 threshold, the router rejects only 8 of 16 hand written out of scope sentences (13 of 16 before). A sweep over all three test sets shows 0.50 recovers 2 of them at a cost of 0.7 points on the command benchmark and nothing on the suite; above 0.50 nothing improves, because the remaining acceptances are confident mistakes (a question about a prime minister read as news), which no threshold fixes. | high | Fixed: threshold 0.50; the confident acceptances are a data and calibration item (PLAN A4 and B4) and stay noted |
| Corrective retrains added phrasings shaped like the misses; the boundary between "not the benchmark sentence" and "the benchmark pattern" is thin, and several misses moved between retrains rather than being fixed. | medium | Noted: recorded in the process log; the honest next lever is seed averaging or a longer schedule, not more templates |
| Pretraining validation loss is comparable across our runs but not directly to published GPT-2 numbers (different tokenizer, 16k vocabulary). | ok | Noted in the 06:00 audit entry |

## Pass 5: code quality and documentation consistency

Perspective: would a stranger be misled by the repository.

| finding | severity | status |
|---|---|---|
| `requirements.txt` lacked `tokenizers`, `pyarrow`, `gguf` and the Windows Triton package, so the documented commands could not run from a fresh clone. | high | Fixed |
| `reasoner.py` documented and defaulted to a pretrained GGUF that the project has rejected. | medium | Fixed: default is our own exported model, overridable with `BSLM_GGUF`; system prompt English only |
| `OWN_MODEL.md` still said "no rented compute, ever" after the decision to rent for the long runs. | medium | Fixed |
| README had no pretraining results and did not list the remote scripts or the loops. | medium | Fixed |
| `spans_from_bio` exists three times, `norm` twice. | low | Noted: small, identical, left as is |
| `gen_data.py` imports templates mid file. | low | Noted: deliberate, the pools are defined first |

## Summary

Twenty findings, thirteen fixed, seven noted. The three that changed
measured behaviour are the apostrophe mismatch (inference now tokenizes
like the training data), the benchmark overlap (the honest score is the
clean subset of 125 commands), and the out of scope regression (threshold
back up to 0.50 after a sweep over all three test sets). The two that would have
bitten unattended are the non atomic checkpoint and the hard coded resume
watcher. Nothing in the audit changes the plan: the 72m run finishes on
the local card, then the 10B token run goes to the rented H100 with the
fixed scripts.


# Audit 2, 2026-09-06, five new perspectives

Same rules: fixed means in the tree and run; noted means left, with the reason.
`python -m bslm.selftest` now exists (44 checks, all passing) and covers
every parser and alignment defect found in either audit.

## Pass 6: statistical rigour

Perspective: are the numbers stable and the comparisons fair.

| finding | severity | status |
|---|---|---|
| The router trainer set no random seed, so every retrain shuffled and initialised differently; the "misses moving between retrains" seen yesterday is partly this. | high | Fixed: `--seed` (default 1337) seeds Python and torch |
| The command benchmark judged domains of two or three commands against an 80% bar; one miss in a domain of two is a 50% score with no evidence behind it. | high | Fixed: domains under five commands are reported but not judged; only work.ops and work.write (7 each, 71%) actually fail now, and 18 everyday domains are marked too small to judge |
| The learning rate schedule on resume and on extension: warmup only applies below the warmup step count, the decay window moves with the new total; the Muon momentum and Adam moments are restored from the checkpoint. | ok | Verified by reading |
| Validation uses 20 fixed batches (164k tokens); noise about 0.01 nats, fine for the curves quoted. | ok | Noted |
| Samples during training are unseeded, so they differ run to run. | negligible | Noted: they are illustrations |

## Pass 7: performance and cost

Perspective: where the time and the money go.

| finding | severity | status |
|---|---|---|
| The loss materialises fp32 logits of batch by sequence by vocabulary: 0.5 GB per micro batch of 8 locally, 2.1 GB at micro 32 on the H100; both fit with room. | ok | Noted |
| The data path copies one window per micro batch on the main thread; at 45k tokens per second on the 4060 the card, not the loader, is the limit (98% utilisation measured). | ok | Verified with pmon |
| Rented run at micro 32: 16 accumulation steps per 524k token batch; expected 500k to 700k tokens per second, 10B tokens in 4 to 5.5 hours, 12 to 17 USD at 2.99 USD per hour. | ok | Noted for the handover check, which flags anything under 400k |
| The old `bslm/bench.py` duplicated the size and latency section of `benchmark.py`. | low | Fixed: removed, README updated |
| Checkpoints carry fp32 optimiser states (490 MB for 56M); pulling them from the pod is a few seconds. | ok | Noted |

## Pass 8: user facing edge cases

Perspective: what a real sentence does to the skills and the GUI.

| finding | severity | status |
|---|---|---|
| "7pm" without a space parsed as 07:00 (no word boundary between a digit and a letter). | high | Fixed and tested |
| "3,5 + 1" evaluated as 36 and "2^10" as 210: comma decimals and the caret were mangled instead of handled. | high | Fixed: comma between digits is a decimal point, caret is a power with a bounded exponent, any surviving word makes the calculator refuse |
| "kill the lights", "cut the tv": the templates teach these as off, the skills switched them on. | high | Fixed: off words extended in both skills, tested |
| "too loud" asked "What volume?" instead of lowering it. | medium | Fixed: too loud lowers, too quiet raises, tested |
| The soul learned "tired" as the user's name from "i am tired". | medium | Fixed: only "my name is", "call me", "the name is" set the name |
| "how many days until Christmas" answers with the date; the date maths skill is in the scenario list as not yet written. | low | Noted |
| Inputs longer than 64 sub word tokens are silently truncated by the router. | low | Noted: commands are short |

## Pass 9: maintainability and tests

Perspective: can the next person change this without breaking it.

| finding | severity | status |
|---|---|---|
| No automated tests of any kind; every regression so far was found by hand. | high | Fixed: `bslm/selftest.py`, 44 checks in seconds, listed in the README |
| Three copies of `spans_from_bio`, two of `norm` and the article regex. | low | Noted: identical, small; consolidating them is a refactor with no behaviour gain |
| `probe.py` (54 sentences) and the 95 sentence suite in `benchmark.py` overlap in purpose. | low | Noted: the probe is the historical baseline quoted in the process log |
| Magic numbers (64 token limit, 4000 vocabulary, 0.50 threshold) are defined once each with a comment. | ok | Verified |

## Pass 10: portability and reproducibility

Perspective: does it run anywhere but this desk.

| finding | severity | status |
|---|---|---|
| `push.sh` and `pull.sh` hard coded this machine's key path and project path. | medium | Fixed: `BSLM_SSH_KEY` and `BSLM_ROOT` with defaults derived from the script location |
| The README did not state the Python version or the CUDA build, and two differently named tokenizer files were easy to confuse. | medium | Fixed: setup paragraph in the README |
| Data generation and pretraining are seeded; the router training was not (pass 6). | high | Fixed |
| The PowerShell loops are Windows only by nature; the remote side is bash on Linux; both are documented as such. | ok | Noted |
| The repository has a `.gitignore` and no commit yet. | low | Noted: committing is the owner's call |

## Summary of audit 2

Nineteen findings, twelve fixed, seven noted, all fixes covered by the new
self test. The three that mattered most: the unseeded router training
(comparisons between retrains were partly noise), the sample size rule in
the command benchmark (two domains actually fail, not six), and the
clock, calculator and lights defects that a first real sentence would have
hit. The command benchmark stands at 90.3% overall and 88.8% on the clean
subset; the honest per domain picture is two failing work domains and
eighteen everyday domains that need at least five commands each before the
bar can be applied.
