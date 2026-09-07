# The Android shell

Our protocol inside llama.cpp's own Android example app (`examples/llama.android`
of the llama.cpp tree at tag b10809, cloned under `tools/llama.cpp-src`).

- `bslm/BslmEnv.kt`: the model's hands on the phone, same actions and same result
  text as `pretrain/agent_tools.py` (Wikipedia search and pages with infobox rows,
  Open-Meteo, YouTube result pages and intents, the clock app for timers and alarms,
  a small store for reminders, lists, notes and events, maths, units, lights,
  switches, volume, the tool registry from `files/tools.json`).
- `bslm/BslmAgent.kt`: the loop from `bslm/agent.py` (header, Earlier memory, new
  round rule, stop before Result, delivery check).
- `bslm/BslmActivity.kt`: one text box and a log; loads the newest GGUF under the
  app's files/models or external files/models, or a picked file. Test hook:
  `adb shell "am start -n com.example.llama.aichat/com.example.llama.bslm.BslmActivity --es q 'who directed inception'"`
  and read `adb logcat -d bslm:I *:S`.
- `lib/patch_example*.py`: the changes to the example's engine: greedy sampling,
  `resetContext()` and `cancelGeneration()` in the JNI layer and the Kotlin
  interface, INTERNET permission, our activity as the launcher.

Build: JDK 17 in `tools/jdk`, SDK at C:/Android/sdk with build tools 36, CMake
3.31.6 and NDK r27c (set `ndkVersion` in `lib/build.gradle.kts` to the installed
one, and `-DFETCHCONTENT_SOURCE_DIR_KLEIDIAI` to `tools/kleidiai-v1.24.0`), then
`bash ./gradlew assembleDebug --no-daemon` in the example directory. Copy the
three Kotlin files into `app/src/main/java/com/example/llama/bslm/` and apply the
patch scripts first. Push a model with `adb push` to /data/local/tmp and
`run-as com.example.llama.aichat cp ... files/models/`.
