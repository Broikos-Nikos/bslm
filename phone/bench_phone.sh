#!/usr/bin/env bash
# Benchmarks the model on an Android device over adb, no app needed.
#   bash phone/bench_phone.sh                 (phone plugged in with USB debugging on, or an emulator running)
# Uses the static binary for the device's ABI from phone/bin/<abi>/ (built by
# phone/build_android.sh) and the GGUF files from models/. On an emulator the
# numbers only prove that the model runs; speed is measured on a real phone.
set -uo pipefail
export MSYS_NO_PATHCONV=1   # Git Bash on Windows would rewrite /data/local/tmp into a C: path
ROOT="${BSLM_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
ADB="${ADB:-adb}"
DEV=/data/local/tmp/bslm
THREADS="${THREADS:-4}"
cd "$ROOT"
$ADB devices | grep -qE "device$" || { echo "no device attached over adb"; exit 1; }
ABI=$($ADB shell getprop ro.product.cpu.abi | tr -d '\r')
[ -x "phone/bin/$ABI/llama-bench" ] || { echo "no phone/bin/$ABI/llama-bench, run: bash phone/build_android.sh $ABI"; exit 1; }
$ADB shell "getprop ro.product.model; getprop ro.board.platform; grep -m1 Features /proc/cpuinfo; grep -c processor /proc/cpuinfo" | tr -d '\r'
$ADB shell mkdir -p $DEV
$ADB push "phone/bin/$ABI/llama-bench" $DEV/ >/dev/null
$ADB shell chmod +x $DEV/llama-bench
for m in 72m-10b-q8.gguf 72m-10b-q4_0.gguf; do
  $ADB push "models/$m" $DEV/ >/dev/null
  echo "=== $m ==="
  $ADB shell "cd $DEV && ./llama-bench -m $m -t $THREADS -p 256 -n 64" 2>&1 | grep -E "^\|" | tail -2
done
$ADB shell rm -rf $DEV
