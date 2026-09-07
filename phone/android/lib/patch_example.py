"""Android shell: greedy sampling, a context reset in the JNI layer and the
engine interface, INTERNET permission, our activity as the launcher."""
from pathlib import Path

APP = Path(r"C:\xampp\htdocs\playground\bslm\tools\llama.cpp-src\examples\llama.android")


def patch(rel, pairs, must=True):
    p = APP / rel
    s = p.read_text(encoding="utf-8")
    for old, new in pairs:
        if old not in s:
            if must:
                raise SystemExit(f"not found in {rel}: {old[:60]}")
            continue
        s = s.replace(old, new, 1)
    p.write_text(s, encoding="utf-8", newline="\n")
    print("patched", rel)


patch("lib/src/main/cpp/ai_chat.cpp", [
    ("constexpr float DEFAULT_SAMPLER_TEMP    = 0.3f;",
     "constexpr float DEFAULT_SAMPLER_TEMP    = 0.0f;   // greedy: the protocol lines are copied, not sampled"),
])
cpp = APP / "lib/src/main/cpp/ai_chat.cpp"
s = cpp.read_text(encoding="utf-8")
if "InferenceEngineImpl_resetContext" not in s:
    s += '''
// BSLM: start a fresh episode without unloading the model. The runtime
// rebuilds the context every turn (header, earlier turns, the user's line).
extern "C"
JNIEXPORT jint JNICALL
Java_com_arm_aichat_internal_InferenceEngineImpl_resetContext(JNIEnv * /*env*/, jobject /*unused*/) {
    reset_long_term_states();
    reset_short_term_states();
    return 0;
}
'''
    cpp.write_text(s, encoding="utf-8", newline="\n")
    print("resetContext added to ai_chat.cpp")

patch("lib/src/main/java/com/arm/aichat/InferenceEngine.kt", [
    ("    fun cleanUp()",
     "    /** BSLM: clear the context, keep the model; a system prompt may follow again. */\n    suspend fun resetContext()\n\n    fun cleanUp()"),
])
patch("lib/src/main/java/com/arm/aichat/internal/InferenceEngineImpl.kt", [
    ("    private external fun shutdown()",
     "    private external fun shutdown()\n\n    private external fun resetContext(): Int"),
    ("    override fun cleanUp() {",
     '''    override suspend fun resetContext() =
        withContext(llamaDispatcher) {
            _cancelGeneration = false
            resetContext()
            _readyForSystemPrompt = true
            _state.value = InferenceEngine.State.ModelReady
        }

    override fun cleanUp() {'''),
])
patch("app/src/main/AndroidManifest.xml", [
    ('<manifest xmlns:android="http://schemas.android.com/apk/res/android">',
     '<manifest xmlns:android="http://schemas.android.com/apk/res/android">\n\n    <uses-permission android:name="android.permission.INTERNET" />\n    <uses-permission android:name="android.permission.READ_MEDIA_AUDIO" />'),
    ('''        <activity
            android:name=".MainActivity"
            android:exported="true">
            <intent-filter>
                <action android:name="android.intent.action.MAIN" />

                <category android:name="android.intent.category.LAUNCHER" />
            </intent-filter>
        </activity>''',
     '''        <activity
            android:name=".bslm.BslmActivity"
            android:exported="true">
            <intent-filter>
                <action android:name="android.intent.action.MAIN" />

                <category android:name="android.intent.category.LAUNCHER" />
            </intent-filter>
        </activity>
        <activity
            android:name=".MainActivity"
            android:exported="false" />'''),
])
print("done")
