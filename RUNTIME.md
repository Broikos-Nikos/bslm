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
- **Quantisation per device:** Q8_0 on the PC (no quality loss, 77.7 MB),
  measure Q4_0 against Q8_0 on the phone, where the Arm kernels make Q4_0
  faster and the 45 MB file leaves room; keep Q8_0 if the benchmark shows
  quality loss on the command suite.

## Expected numbers

| device | prompt tokens per second | generation tokens per second | one loop step of 500 tokens |
|---|---|---|---|
| RTX 4060, CUDA | several thousand | over 1,000 | well under a second |
| PC, 4 CPU threads (measured) | 1,858 | 298 | about half a second |
| recent Snapdragon, 4 big cores, Q4_0 with KleidiAI (estimate) | 500 to 1,000 | 80 to 150 | one to two seconds |

The phone estimate is the one to replace with a measurement before any
app code is written: install Termux, copy `llama-bench` and the two GGUF
files, run `llama-bench -m model.gguf -t 4 -p 256 -n 64`, and the table
above becomes fact for that exact phone.

## Order of work

1. Measure on the phone with `llama-bench` (Q8_0 and Q4_0), ten minutes.
2. Build the llama.cpp Android example with our GGUF, confirm it runs, keep
   the JNI layer, throw away the example UI.
3. Move the loop and the tool registry into a shared core once the Python
   version on the PC stops changing.
4. Only if the phone numbers disappoint: ExecuTorch with the QNN delegate for
   the NPU, at the cost of a second export path.
