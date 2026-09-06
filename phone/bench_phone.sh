#!/usr/bin/env bash
# Benchmarks the model on an Android phone over adb, no app needed.
#   bash phone/bench_phone.sh                     (phone plugged in, USB debugging on)
# phone/bin/ holds llama-bench and its shared libraries from the llama.cpp
# Android arm64 release (b10809); the GGUF files come from models/.
set -uo pipefail
export MSYS_NO_PATHCONV=1   # Git Bash on Windows would rewrite /data/local/tmp into a C: path
ROOT="${BSLM_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
ADB="${ADB:-adb}"
DEV=/data/local/tmp/bslm
cd "$ROOT"
$ADB devices | grep -qE "device$" || { echo "no phone attached over adb"; exit 1; }
$ADB shell getprop ro.product.model; $ADB shell getprop ro.board.platform
$ADB shell "grep -m1 Features /proc/cpuinfo; grep -c processor /proc/cpuinfo"
$ADB shell mkdir -p $DEV
for f in phone/bin/*; do $ADB push "$f" $DEV/ >/dev/null; done
$ADB shell chmod +x $DEV/llama-bench
for m in 72m-10b-q8.gguf 72m-10b-q4_0.gguf; do
  $ADB push "models/$m" $DEV/ >/dev/null
  echo "=== $m ==="
  $ADB shell "cd $DEV && LD_LIBRARY_PATH=$DEV ./llama-bench -m $m -t 4 -p 256 -n 64" 2>&1 | grep -E "^\|" | tail -2
done
$ADB shell rm -rf $DEV
