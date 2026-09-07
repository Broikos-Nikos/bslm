#!/usr/bin/env bash
# Stage 5 to 7 in one go: trajectories -> fine tune -> GGUF -> agent benchmark.
#   bash pretrain/agent_pipeline.sh [run-name] [init-run]
# Waits for the fact fetch (corpus/agent/facts.log ends with "places:") if it
# is still running. Logs: corpus/agent/gen.log, pretrain/runs/<run>/log.txt,
# corpus/agent/pipeline.log.
set -uo pipefail
ROOT="${BSLM_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
export PYTHONIOENCODING=utf-8
PY=./.venv/Scripts/python.exe
RUN="${1:-72m-agent}"
INIT="${2:-72m-10b}"
LOG=corpus/agent/pipeline.log
say() { echo "$(date +%H:%M:%S) $*" | tee -a "$LOG"; }

say "pipeline start: run $RUN from $INIT"
for i in $(seq 1 120); do
  grep -q "^places:" corpus/agent/facts.log 2>/dev/null && break
  sleep 30
done
grep -q "^places:" corpus/agent/facts.log || { say "fact fetch did not finish, stopping"; exit 1; }
say "facts: $(wc -l < corpus/agent/facts.jsonl) songs: $(wc -l < corpus/agent/songs.jsonl) places: $(wc -l < corpus/agent/places.jsonl)"

if [ "${SKIP_GEN:-0}" = "1" ]; then
  # a generator started earlier is still running: wait for its last line
  say "waiting for the running generator"
  for i in $(seq 1 720); do grep -q "^test:" corpus/agent/gen.log 2>/dev/null && break; sleep 30; done
  grep -q "^test:" corpus/agent/gen.log || { say "generator did not finish"; exit 1; }
else
  say "trajectories"
  $PY -u -m pretrain.trajectories --facts "${FACTS:-20000}" --songs "${SONGS:-3000}" --local "${LOCAL:-14000}" \
      --compound "${COMPOUND:-3000}" --umbrella "${UMBRELLA:-1500}" --other "${OTHER:-1000}" --workers 8 \
      --phrasings "${PHRASINGS:-1}" --giveup_keep "${GIVEUP_KEEP:-0.15}" \
      > corpus/agent/gen.log 2>&1 || { say "generator failed, see corpus/agent/gen.log"; tail -5 corpus/agent/gen.log; exit 1; }
fi
tail -4 corpus/agent/gen.log | tee -a "$LOG"

say "fine tune"
$PY -u -m pretrain.sft --name "$RUN" --init "$INIT" --epochs "${EPOCHS:-3}" > "corpus/agent/sft_$RUN.log" 2>&1 \
    || { say "fine tune failed, see corpus/agent/sft_$RUN.log"; tail -5 "corpus/agent/sft_$RUN.log"; exit 1; }
grep -E "^eval|^done" "pretrain/runs/$RUN/log.txt" | tail -3 | tee -a "$LOG"

say "export"
$PY -m pretrain.export_gguf --name "$RUN" >> "$LOG" 2>&1 || { say "export failed"; exit 1; }
./tools/llama/llama-quantize.exe "models/$RUN-f16.gguf" "models/$RUN-q8.gguf" Q8_0 > /dev/null 2>&1 || { say "quantize failed"; exit 1; }
say "gguf: $(du -m models/$RUN-q8.gguf | cut -f1) MB"

say "agent benchmark"
BSLM_GGUF="models/$RUN-q8.gguf" BSLM_LLM_THREADS="${THREADS:-8}" $PY -u -m bslm.agent_bench --limit "${LIMIT:-100}" > corpus/agent/bench.log 2>&1 \
    || { say "benchmark failed, see corpus/agent/bench.log"; tail -5 corpus/agent/bench.log; exit 1; }
tail -2 corpus/agent/bench.log | tee -a "$LOG"
say "pipeline done"
