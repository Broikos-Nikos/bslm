#!/usr/bin/env bash
# One round of the loop model, in separable stages so generation of the next
# round can overlap the training of this one (generation is network and CPU,
# training is the GPU; never two trainings at once).
#
#   bash pretrain/agent_round.sh <run> <init> gen      -> corpus/agent/rounds/<run>/{train,val}.npy, test.jsonl
#   bash pretrain/agent_round.sh <run> <init> train    -> fine tune, GGUF, benchmark, from that directory
#   bash pretrain/agent_round.sh <run> <init> all      -> both, in sequence
#   DATA=corpus/agent/rounds/72m-agent7 bash pretrain/agent_round.sh 56m-agent8 56m-fineweb train
#                                                      -> train another model on an existing round's data
#
# Knobs (environment): FACTS PHRASINGS LOCAL COMPOUND UMBRELLA OTHER FOLLOWUP GIVEUP_KEEP
# for generation; EPOCHS MICRO LR_MUON LR_ADAM for training; LIMIT THREADS for the
# benchmark. Logs: <data>/gen.log, corpus/agent/sft_<run>.log, corpus/agent/bench_<run>.log,
# and one line per stage in corpus/agent/pipeline.log.
set -uo pipefail
ROOT="${BSLM_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
export PYTHONIOENCODING=utf-8
PY=./.venv/Scripts/python.exe
RUN="${1:?run name}"
INIT="${2:-72m-10b}"
STAGE="${3:-all}"
DATA="${DATA:-corpus/agent/rounds/$RUN}"
LOG=corpus/agent/pipeline.log
say() { echo "$(date +%H:%M:%S) [$RUN] $*" | tee -a "$LOG"; }

gen() {
  mkdir -p "$DATA"
  say "generate -> $DATA"
  $PY -u -m pretrain.trajectories --facts "${FACTS:-16000}" --songs "${SONGS:-3000}" --local "${LOCAL:-8000}" \
      --compound "${COMPOUND:-3000}" --umbrella "${UMBRELLA:-1500}" --other "${OTHER:-1000}" --workers 8 \
      --phrasings "${PHRASINGS:-2}" --giveup_keep "${GIVEUP_KEEP:-0.25}" --followup "${FOLLOWUP:-3000}" \
      --out "$DATA" > "$DATA/gen.log" 2>&1 || { say "generator failed, see $DATA/gen.log"; tail -5 "$DATA/gen.log"; return 1; }
  tail -3 "$DATA/gen.log" | tee -a "$LOG"
}

train() {
  [ -f "$DATA/train.npy" ] || { say "no data in $DATA"; return 1; }
  say "fine tune from $INIT on $DATA (epochs ${EPOCHS:-3})"
  $PY -u -m pretrain.sft --name "$RUN" --init "$INIT" --epochs "${EPOCHS:-3}" --micro "${MICRO:-8}" \
      --lr_muon "${LR_MUON:-0.004}" --lr_adam "${LR_ADAM:-4e-4}" --data "$DATA" > "corpus/agent/sft_$RUN.log" 2>&1 \
      || { say "fine tune failed, see corpus/agent/sft_$RUN.log"; tail -5 "corpus/agent/sft_$RUN.log"; return 1; }
  grep -E "^eval|^done" "pretrain/runs/$RUN/log.txt" | tail -2 | tee -a "$LOG"
  say "export"
  $PY -m pretrain.export_gguf --name "$RUN" >> "$LOG" 2>&1 || { say "export failed"; return 1; }
  ./tools/llama/llama-quantize.exe "models/$RUN-f16.gguf" "models/$RUN-q8.gguf" Q8_0 > /dev/null 2>&1 || { say "quantize failed"; return 1; }
  say "benchmark"
  BSLM_GGUF="models/$RUN-q8.gguf" BSLM_LLM_THREADS="${THREADS:-8}" $PY -u -m bslm.agent_bench --limit "${LIMIT:-100}" --data "$DATA" \
      > "corpus/agent/bench_$RUN.log" 2>&1 || { say "benchmark failed, see corpus/agent/bench_$RUN.log"; tail -5 "corpus/agent/bench_$RUN.log"; return 1; }
  cp AGENT_BENCHMARK.md "corpus/agent/rounds/AGENT_BENCHMARK_$RUN.md" 2>/dev/null
  tail -2 "corpus/agent/bench_$RUN.log" | tee -a "$LOG"
}

case "$STAGE" in
  gen) gen ;;
  train) train ;;
  all) gen && train ;;
  *) echo "stage must be gen, train or all"; exit 1 ;;
esac
say "stage $STAGE done"
