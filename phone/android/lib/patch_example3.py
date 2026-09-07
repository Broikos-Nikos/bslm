from pathlib import Path

APP = Path(r"C:\xampp\htdocs\playground\bslm\tools\llama.cpp-src\examples\llama.android")


def patch(rel, pairs):
    p = APP / rel
    s = p.read_text(encoding="utf-8")
    for old, new in pairs:
        if old not in s:
            raise SystemExit(f"not found in {rel}: {old[:60]}")
        s = s.replace(old, new, 1)
    p.write_text(s, encoding="utf-8", newline="\n")
    print("patched", rel)


patch("lib/src/main/java/com/arm/aichat/internal/InferenceEngineImpl.kt", [
    ("    private external fun resetContext(): Int", "    private external fun nativeResetContext(): Int"),
    ('''            _cancelGeneration = false
            resetContext()
            _readyForSystemPrompt = true''',
     '''            _cancelGeneration = false
            nativeResetContext()
            _readyForSystemPrompt = true'''),
])
patch("lib/src/main/cpp/ai_chat.cpp", [
    ("Java_com_arm_aichat_internal_InferenceEngineImpl_resetContext(", "Java_com_arm_aichat_internal_InferenceEngineImpl_nativeResetContext("),
])
print("done")
