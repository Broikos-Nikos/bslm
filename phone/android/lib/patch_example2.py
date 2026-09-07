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


patch("lib/src/main/java/com/arm/aichat/InferenceEngine.kt", [
    ("    suspend fun resetContext()",
     "    suspend fun resetContext()\n\n    /** BSLM: stop the current generation; the next prompt continues the context. */\n    fun cancelGeneration()"),
])
patch("lib/src/main/java/com/arm/aichat/internal/InferenceEngineImpl.kt", [
    ("    override suspend fun resetContext() =",
     "    override fun cancelGeneration() {\n        _cancelGeneration = true\n    }\n\n    override suspend fun resetContext() ="),
    ('''            _readyForSystemPrompt = false
            _state.value = InferenceEngine.State.ProcessingUserPrompt''',
     '''            _readyForSystemPrompt = false
            _cancelGeneration = false
            _state.value = InferenceEngine.State.ProcessingUserPrompt'''),
])
print("done")
