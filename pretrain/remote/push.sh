#!/usr/bin/env bash
# From the local machine: sync code and token shards to a pod over SSH.
#   bash pretrain/remote/push.sh <pod-ip> <ssh-port>
# Uses the key registered on the RunPod account. Shards go to the pod's
# /workspace (put a network volume there so they survive pod restarts).
set -euo pipefail
IP="$1"; PORT="${2:-22}"
KEY="/c/Users/Administrator/.ssh/runpod_bslm"
SSH="ssh -i $KEY -p $PORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
cd /c/xampp/htdocs/playground/bslm
$SSH root@$IP "mkdir -p /workspace/bslm/pretrain /workspace/bslm/bslm /workspace/corpus/tokens"
# --no-same-owner: a Windows tar carries uids the pod cannot apply, and root
# refusing to chown would otherwise abort the whole script under set -e
tar cf - pretrain/*.py pretrain/__init__.py pretrain/tokenizer.json pretrain/remote bslm/__init__.py | $SSH root@$IP "cd /workspace/bslm && tar xf - --no-same-owner"
# shards: 4.6 GB, resumable with rsync if available, else scp
if command -v rsync >/dev/null 2>&1; then
  rsync -av --progress -e "$SSH" corpus/tokens/ root@$IP:/workspace/corpus/tokens/
else
  scp -i "$KEY" -P "$PORT" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null corpus/tokens/*.npy corpus/tokens/meta.json root@$IP:/workspace/corpus/tokens/
fi
$SSH root@$IP "ls /workspace/corpus/tokens | wc -l; du -sh /workspace/corpus/tokens"
