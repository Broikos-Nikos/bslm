#!/usr/bin/env bash
# Copies the token shards to a pod one file at a time, skipping files that
# already have the right size on the pod, so it can be rerun after any
# interruption. Meant to be launched detached (Start-Process) on the local
# machine, because the harness kills long background jobs when RAM is low.
#   bash pretrain/remote/push_shards.sh <pod-ip> <ssh-port>
set -uo pipefail
IP="$1"; PORT="${2:-22}"
KEY="${BSLM_SSH_KEY:-$HOME/.ssh/runpod_bslm}"
ROOT="${BSLM_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
LOG="$ROOT/pretrain/runs/push_shards.log"
SSH="ssh -i $KEY -p $PORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"
cd "$ROOT"
echo "=== push start $(date +%H:%M:%S) ===" >> "$LOG"
$SSH root@$IP "mkdir -p /workspace/corpus/tokens"
for f in corpus/tokens/*.npy corpus/tokens/meta.json; do
  name=$(basename "$f")
  local_size=$(stat -c%s "$f")
  remote_size=$($SSH root@$IP "stat -c%s /workspace/corpus/tokens/$name 2>/dev/null || echo 0")
  if [ "$remote_size" = "$local_size" ]; then
    echo "skip $name (present)" >> "$LOG"
    continue
  fi
  echo "copy $name ($local_size bytes) $(date +%H:%M:%S)" >> "$LOG"
  scp -i "$KEY" -P "$PORT" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR "$f" "root@$IP:/workspace/corpus/tokens/$name" >> "$LOG" 2>&1 || echo "FAILED $name" >> "$LOG"
done
echo "=== push done $(date +%H:%M:%S) $($SSH root@$IP 'ls /workspace/corpus/tokens | wc -l') files ===" >> "$LOG"
