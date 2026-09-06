#!/usr/bin/env bash
# Runs INSIDE a RunPod pod (PyTorch template). Prepares the environment and
# starts a training run detached so it survives the SSH session.
#   bash bootstrap.sh <run-name> <size> <tokens>      e.g. bash bootstrap.sh 72m-10b 72m 1e10
# Expects /workspace/bslm (code) and /workspace/corpus/tokens (shards) to exist,
# both synced from the local machine by pretrain/remote/push.sh.
set -euo pipefail
NAME="${1:-72m-remote}"; SIZE="${2:-72m}"; TOKENS="${3:-1e10}"
cd /workspace/bslm
# the RunPod image is a PEP 668 "externally managed" python: plain pip refuses
pip install -q --break-system-packages tokenizers pyarrow numpy 2>&1 | grep -viE "warning|notice" | tail -1 || true
python -c "import tokenizers, pyarrow" || { echo "tokenizer libraries missing"; exit 1; }
python -c "import torch, triton; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
mkdir -p corpus && ln -sfn /workspace/corpus/tokens corpus/tokens
mkdir -p pretrain/runs
CMD="cd /workspace/bslm && python -m pretrain.train --name $NAME --size $SIZE --tokens $TOKENS --micro 32 --eval_every 200 --ckpt_every 300 --resume"
if command -v tmux >/dev/null 2>&1; then
  tmux kill-session -t train 2>/dev/null || true
  tmux new-session -d -s train "$CMD 2>&1 | tee -a pretrain/runs/$NAME.console.log"
  echo "started run $NAME in tmux session 'train'"
else
  # no tmux on this image: detach with nohup + setsid, same log
  nohup setsid bash -c "$CMD" >> "pretrain/runs/$NAME.console.log" 2>&1 &
  echo "started run $NAME with nohup (pid $!)"
fi
sleep 5
echo "tail with: tail -f /workspace/bslm/pretrain/runs/$NAME/log.txt"
