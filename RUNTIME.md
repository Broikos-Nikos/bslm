# Running the model on the PC and on Android

What the runtime has to carry: a 72M parameter Llama shaped decoder, 77.7 MB
at Q8_0 (about 45 MB at Q4_0), context 1024, and the loop from
`OWN_MODEL.md` stage 6, which means prompts of 300 to 600 tokens per step and
short generations. Prompt processing speed matters more than generation
speed, and the tools around the model (search, calendar, timers) are
platform code, not model code.

## Options, 2026

| runtime | Windows | Android | performance on a 72M model | effort | verdict |
|---|---|---|---|---|---|
| **llama.cpp** (C++, GGUF) | CPU, CUDA, Vulkan builds; `llama-server` or direct embedding | NDK build, official `examples/llama.android`, JNI from Kotlin, KleidiAI and SME2 kernels for Arm | best in class for GGUF; measured here 1,858 prompt and 298 generation tokens per second on 4 CPU threads, thousands per second on the 4060 | medium: one C API, two thin shells | **the core, on both** |
| ExecuTorch (PyTorch export to `.pte`) | weak | XNNPACK on CPU, Vulkan on GPU, QNN on the Snapdragon NPU | the only path to the phone's NPU; for a model this small the export and delegate overhead buys little | high | keep for later if the phone's NPU is ever needed |
| MLC LLM (TVM compiled) | Vulkan | OpenCL and Vulkan GPU | strong for 1B and up; a 72M model is launch bound on a phone GPU and gains nothing | high | no |
| ONNX Runtime GenAI | DirectML | NNAPI, QNN | generalist, slower than llama.cpp on LLM shapes, needs an ONNX export and its own tokenizer plumbing | medium | no |
| MediaPipe LLM Inference | no | fixed architectures only | cannot load a custom model | | no |
| Rust core (`llama-cpp-rs` or candle) shared through UniFFI or JNI | native | native | llama.cpp speed with one codebase for the loop logic | medium to high | the way to share the brain code if the shells grow |
| React Native or Flutter with a llama.cpp plugin (`llama.rn`, cactus) | works | works | near native inference, UI in one codebase | medium | only if a cross platform UI matters more than the last 10% |

Sources read for this table: the llama.cpp Android docs and performance
discussions, Arm's KleidiAI and SME2 guide for llama.cpp, comparisons of
llama.cpp, ExecuTorch, MLC LLM and ONNX Runtime GenAI for on device use.

## Recommendation

**llama.cpp as the single inference core, the same GGUF file on both
devices, and two thin shells.**

- **PC (Windows):** `llama-server` as a local sidecar (what `bslm/reasoner.py`
  already talks to), CUDA build for the 4060 or the CPU build when the card
  is busy. The loop and the tools stay in Python until they are stable, then
  move into the shared core below.
- **Android:** a Kotlin app with a JNI bridge to llama.cpp built with
  `-DGGML_CPU_KLEIDIAI=ON`, the model loaded memory mapped from the app's
  files directory, four big cores, batch prefill for the prompt. Start from
  the official `examples/llama.android` project; it is the shell, the brain
  is ours.
- **Shared brain:** once the loop (plan, act, observe, judge, retry, deliver)
  is settled, implement it once in a small library with a C interface (C++
  next to llama.cpp, or Rust with UniFFI) so the PC service and the Android
  app run identical logic; tools are registered by each platform.
- **Quantisation, decided by measurement on held out text (llama-perplexity,
  40 chunks of 1024 tokens from the validation shard):** f16 17.30, Q8_0
  17.32, Q4_0 18.78. Q8_0 is lossless within noise; Q4_0 costs 8.5% in
  perplexity, which the owner does not accept. **Q8_0 everywhere, 77.7 MB,
  on the PC and on every phone.** Speed on the phone comes from the free
  levers below, not from a smaller number format.
- **No NPU dependence.** The target is the owner's Poco F3 (Snapdragon 870,
  2021) today and weaker 2026 phones tomorrow, so the CPU path with
  llama.cpp's Arm kernels is the only path; NPU delegates would fragment
  the app per chip family and are off the table.

## Expected numbers

| device | prompt tokens per second | generation tokens per second | one loop step of 500 tokens |
|---|---|---|---|
| RTX 4060, CUDA | several thousand | over 1,000 | well under a second |
| PC, 4 CPU threads (measured) | 1,858 | 298 | about half a second |
| Poco F3, Snapdragon 870, 4 big cores, Q8_0 (estimate) | 500 to 1,000 | 80 to 150 | one to two seconds |

The phone estimate is the one to replace with a measurement before any
app code is written. `phone/bench_phone.sh` does it over adb with the
prebuilt Android arm64 `llama-bench` from the llama.cpp release (staged in
`phone/bin/`, not committed): plug the phone in with USB debugging on and
the script pushes the binary and the GGUF files to `/data/local/tmp`, runs
the benchmark on 4 threads, prints the numbers and cleans up.

## Order of work

1. Measure on the phone with `phone/bench_phone.sh` (Q8_0, and Q4_0 for reference), ten minutes.
2. Build the llama.cpp Android example with our GGUF, confirm it runs, keep
   the JNI layer, throw away the example UI.
3. Move the loop and the tool registry into a shared core once the Python
   version on the PC stops changing.
4. Only if the phone numbers disappoint: ExecuTorch with the QNN delegate for
   the NPU, at the cost of a second export path.
