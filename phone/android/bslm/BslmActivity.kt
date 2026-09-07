package com.example.llama.bslm

import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.util.TypedValue
import android.view.inputmethod.EditorInfo
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.arm.aichat.AiChat
import com.arm.aichat.InferenceEngine
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONArray
import java.io.File
import java.io.FileOutputStream

/**
 * The phone shell: one text box, one log. The model file is the first GGUF
 * under the app's external files directory (adb push works there) or one
 * picked from storage. A test hook: `adb shell am start -n <pkg>/.bslm.BslmActivity
 * --es q "who directed inception"` runs one turn and prints the outcome to logcat.
 */
class BslmActivity : AppCompatActivity() {
    private lateinit var log: TextView
    private lateinit var input: EditText
    private lateinit var scroll: ScrollView
    private lateinit var engine: InferenceEngine
    private var agent: BslmAgent? = null
    private var pendingAsk = false

    private val pick = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri -> uri?.let { importModel(it) } }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val root = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(24, 48, 24, 24) }
        log = TextView(this).apply { setTextSize(TypedValue.COMPLEX_UNIT_SP, 15f); setTextIsSelectable(true) }
        scroll = ScrollView(this).apply { addView(log) }
        root.addView(scroll, LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f))
        val row = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        input = EditText(this).apply { hint = "loading model..."; isEnabled = false; imeOptions = EditorInfo.IME_ACTION_SEND; setSingleLine() }
        row.addView(input, LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f))
        val send = Button(this).apply { text = "send"; setOnClickListener { submit() } }
        row.addView(send)
        root.addView(row)
        setContentView(root)
        input.setOnEditorActionListener { _, _, _ -> submit(); true }

        engine = AiChat.getInferenceEngine(this)
        val env = BslmEnv(this, loadTools())
        lifecycleScope.launch(Dispatchers.IO) {
            val model = findModel()
            if (model == null) {
                withContext(Dispatchers.Main) { say("bslm", "pick a GGUF file"); pick.launch(arrayOf("*/*")) }
                return@launch
            }
            loadAndReady(model, env)
        }
    }

    private suspend fun loadAndReady(model: File, env: BslmEnv) {
        try {
            engine.loadModel(model.path)
            agent = BslmAgent(engine, env)
            withContext(Dispatchers.Main) { say("bslm", "ready (${model.name})"); input.isEnabled = true; input.hint = "ask or command" }
            intent?.getStringExtra("q")?.let { q -> handle(q) }
        } catch (e: Exception) {
            Log.e(TAG, "load failed", e)
            withContext(Dispatchers.Main) { say("bslm", "could not load ${model.name}: ${e.message}") }
        }
    }

    private fun findModel(): File? {
        val dirs = listOfNotNull(getExternalFilesDir("models"), File(filesDir, "models"))
        return dirs.flatMap { d -> d.listFiles { f -> f.name.endsWith(".gguf") }?.toList() ?: emptyList() }.maxByOrNull { it.lastModified() }
    }

    private fun importModel(uri: Uri) {
        lifecycleScope.launch(Dispatchers.IO) {
            val dir = File(filesDir, "models").apply { mkdirs() }
            val out = File(dir, "picked.gguf")
            contentResolver.openInputStream(uri)?.use { i -> FileOutputStream(out).use { i.copyTo(it) } }
            loadAndReady(out, BslmEnv(this@BslmActivity, loadTools()))
        }
    }

    private fun loadTools(): List<BslmEnv.Tool> {
        val f = File(getExternalFilesDir(null), "tools.json")
        if (!f.exists()) return emptyList()
        return try {
            val arr = JSONArray(f.readText())
            (0 until arr.length()).map { i -> val o = arr.getJSONObject(i); BslmEnv.Tool(o.getString("name"), o.optString("desc"), o.optString("cmd").ifBlank { null }) }
        } catch (e: Exception) { emptyList() }
    }

    private fun submit() {
        val t = input.text.toString().trim()
        if (t.isEmpty() || agent == null) return
        input.setText("")
        lifecycleScope.launch(Dispatchers.IO) { handle(t) }
    }

    private suspend fun handle(text: String) {
        val a = agent ?: return
        withContext(Dispatchers.Main) { say("you", text); input.isEnabled = false }
        val o = try { if (pendingAsk) a.reply(text) else a.run(text) } catch (e: Exception) { BslmAgent.Outcome("fail", "error: ${e.message}", "", emptyList()) }
        pendingAsk = o.kind == "ask"
        Log.i(TAG, "q=$text kind=${o.kind} answer=${o.answer} acts=${o.trace.map { it.first }}")
        withContext(Dispatchers.Main) {
            for ((act, res) in o.trace) say("  >", "$act\n    ${res.take(200)}")
            say("bslm", (if (o.kind == "ask") "? " else "") + o.answer)
            input.isEnabled = true
        }
    }

    private fun say(who: String, text: String) {
        log.append("$who: $text\n")
        scroll.post { scroll.fullScroll(ScrollView.FOCUS_DOWN) }
    }

    override fun onDestroy() {
        super.onDestroy()
        engine.destroy()
    }

    companion object { const val TAG = "bslm" }
}
