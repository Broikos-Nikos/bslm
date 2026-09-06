#!/usr/bin/env bash
# From the local machine: fetch a run's checkpoint, log and samples from a pod.
#   bash pretrain/remote/pull.sh <pod-ip> <ssh-port> <run-name>
set -euo pipefail
IP="$1"; PORT="${2:-22}"; NAME="${3:-72m-remote}"
KEY="${BSLM_SSH_KEY:-$HOME/.ssh/runpod_bslm}"
cd "${BSLM_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
mkdir -p "pretrain/runs/$NAME"
# one scp per file: the remote shell is not guaranteed to expand braces
for f in ckpt.pt log.txt samples.txt; do
  scp -i "$KEY" -P "$PORT" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    "root@$IP:/workspace/bslm/pretrain/runs/$NAME/$f" "pretrain/runs/$NAME/$f" || echo "missing remote $f"
done
ls -la "pretrain/runs/$NAME/"
