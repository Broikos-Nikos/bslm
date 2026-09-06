#!/usr/bin/env bash
# Builds static llama.cpp tools for Android with the NDK, no shared libraries,
# so a single binary runs from /data/local/tmp on a phone or in the emulator.
#   bash phone/build_android.sh arm64-v8a     -> phone/bin/arm64-v8a/{llama-bench,llama-completion}
#   bash phone/build_android.sh x86_64        -> phone/bin/x86_64/...  (emulator, functional tests only)
# Needs tools/llama.cpp-src (git clone of the b10809 tag) and an NDK under $ANDROID_SDK/ndk.
set -euo pipefail
ABI="${1:-arm64-v8a}"
ROOT="${BSLM_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
SDK="${ANDROID_SDK:-/c/Android/sdk}"
NDK="${ANDROID_NDK:-$(ls -d "$SDK"/ndk/* | tail -1)}"
SRC="$ROOT/tools/llama.cpp-src"
BUILD="$SRC/build-android-$ABI"
if command -v cygpath >/dev/null; then   # Git Bash: cmake is a Windows program, give it Windows paths
  SRC=$(cygpath -m "$SRC"); BUILD=$(cygpath -m "$BUILD"); NDK=$(cygpath -m "$NDK")
fi
JOBS="${JOBS:-2}"                       # low on purpose, the machine is shared
FLAGS=(-DANDROID_ABI="$ABI" -DANDROID_PLATFORM=android-28
       -DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake"
       -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_STATIC=ON
       -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
       -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TOOLS=ON
       -DGGML_OPENMP=OFF -DGGML_LLAMAFILE=OFF -DGGML_NATIVE=OFF)
if [ "$ABI" = "arm64-v8a" ]; then
  # Snapdragon 870 class and up: armv8.2 with dotprod and fp16, no i8mm, so
  # the same binary also runs on weaker 2026 phones; KleidiAI picks the
  # dotprod kernels for Q8_0 and Q4_0.
  FLAGS+=(-DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16 -DGGML_CPU_KLEIDIAI=ON)
fi
cmake -S "$SRC" -B "$BUILD" -G Ninja "${FLAGS[@]}" >"$BUILD.configure.log" 2>&1 || { tail -20 "$BUILD.configure.log"; exit 1; }
cmake --build "$BUILD" -j"$JOBS" --target llama-bench llama-completion >"$BUILD.build.log" 2>&1 || { tail -30 "$BUILD.build.log"; exit 1; }
mkdir -p "$ROOT/phone/bin/$ABI"
cp "$BUILD/bin/llama-bench" "$BUILD/bin/llama-completion" "$ROOT/phone/bin/$ABI/"
"$NDK/toolchains/llvm/prebuilt/windows-x86_64/bin/llvm-strip" "$ROOT/phone/bin/$ABI"/llama-*   # 120 MB unstripped, a few MB stripped
ls -la "$ROOT/phone/bin/$ABI" | awk 'NR>3{printf "%s %.1f MB\n", $9, $5/1e6}'
