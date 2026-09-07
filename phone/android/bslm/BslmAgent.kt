package com.example.llama.bslm

import com.arm.aichat.InferenceEngine
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.flow.takeWhile
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * The trained loop on the phone, the same turn taking as bslm/agent.py: the
 * model owns Plan, Act, Judge, Ask and Deliver; generation stops at the first
 * "Result:" it would write, the environment runs the action for real and the
 * text continues. A light memory of the round (one Earlier line per turn),
 * a hard new round rule, the tools line, and the delivery check.
 */
class BslmAgent(
    private val engine: InferenceEngine,
    val env: BslmEnv,
    var home: String = "Athens",
    var soul: String = "rides a scooter",
    var check: Boolean = true,
) {
    data class Turn(val user: String, val act: String?, val said: String)
    data class Outcome(val kind: String, val answer: String, val rawAnswer: String, val trace: List<Pair<String, String>>, val unverified: Boolean = false)

    val memory = mutableListOf<Turn>()
    private var lastTime = 0L
    private var transcript = StringBuilder()

    fun newRound() { memory.clear() }

    private fun header(): String {
        val now = SimpleDateFormat("EEEE yyyy-MM-dd, HH:mm", Locale.ENGLISH).format(Date())
        val sb = StringBuilder("Today is $now. Home: $home. Facts: $soul.")
        if (env.tools.isNotEmpty()) sb.append("\nTools: ").append(env.tools.joinToString("; ") { "${it.name} (${it.desc})" })
        for (t in memory.takeLast(MEMORY_TURNS)) sb.append("\nEarlier: \"${t.user}\" -> ${t.act ?: "no action"} -> \"${t.said.take(90)}\"")
        return sb.append("\n").toString()
    }

    /** Decode text into the running context and collect what the model writes until a stop marker. */
    private suspend fun complete(text: String, maxTokens: Int = 160): String {
        val out = StringBuilder()
        var stopped = false
        engine.sendUserPrompt(text, maxTokens).takeWhile { !stopped }.collect { piece ->
            out.append(piece)
            if (STOPS.any { out.contains(it) }) stopped = true
        }
        if (stopped) engine.cancelGeneration()
        var s = out.toString()
        for (st in STOPS) { val i = s.indexOf(st); if (i >= 0) s = s.substring(0, i) }
        return s
    }

    suspend fun run(text0: String, maxActs: Int = 6, resume: Boolean = false): Outcome {
        val text = text0.trim()
        val now = System.currentTimeMillis()
        if (!resume) {
            if (lastTime > 0 && now - lastTime > ROUND_GAP_MS) newRound()
            if (NEW_ROUND.matches(text)) { newRound(); lastTime = now; return Outcome("deliver", "Fresh start.", "Fresh start.", emptyList()) }
            engine.resetContext()
            engine.setSystemPrompt(header())
            transcript = StringBuilder()
        }
        var pending = "User: $text\n"
        transcript.append(pending)
        val trace = mutableListOf<Pair<String, String>>()
        repeat(maxActs + 2) {
            val out = complete(pending)
            transcript.append(out)
            val kept = mutableListOf<String>()
            var action: String? = null
            for (l in out.split("\n")) {
                val s = l.trim()
                if (s.isEmpty()) continue
                kept.add(s)
                if (s.startsWith("Deliver:")) return deliver(text, s.removePrefix("Deliver:").trim(), trace, now)
                if (s.startsWith("Ask:")) { lastTime = now; val q = s.removePrefix("Ask:").trim(); return Outcome("ask", q, q, trace) }
                if (s.startsWith("Act:")) { action = s.removePrefix("Act:").trim(); break }
            }
            if (action == null) {
                if (kept.isEmpty()) return fail(trace, now)
                pending = "\n"
                return@repeat
            }
            val result = env.act(action)
            trace.add(action to result)
            pending = "\nResult:\n${result.trim()}\n"
            transcript.append(pending)
            if (action.startsWith("ask(")) { lastTime = now; val q = action.removePrefix("ask(").removeSuffix(")").trim('"', '\'', ' '); return Outcome("ask", q, q, trace) }
        }
        return fail(trace, now)
    }

    private fun fail(trace: List<Pair<String, String>>, now: Long): Outcome {
        lastTime = now
        return Outcome("fail", "I could not finish that.", "I could not finish that.", trace)
    }

    private fun deliver(text: String, raw: String, trace: List<Pair<String, String>>, now: Long): Outcome {
        var answer = raw
        var unverified = false
        if (check && trace.any { it.first.startsWith("search(") || it.first.startsWith("open(") } && !Regex("could not|cannot|can't", RegexOption.IGNORE_CASE).containsMatchIn(raw)) {
            val cand = ANSWER_CUT.find(raw)?.groupValues?.get(1)?.trim() ?: ""
            val last = trace.lastOrNull()?.second ?: ""
            if (cand.isNotEmpty() && (!BslmEnv.norm(last).contains(BslmEnv.norm(cand)) || BslmEnv.norm(text).contains(BslmEnv.norm(cand)))) {
                unverified = true
                answer = "I could not confirm that from the page. What the model read: $raw"
            }
        }
        memory.add(Turn(text, trace.lastOrNull()?.first, answer))
        while (memory.size > MEMORY_TURNS) memory.removeAt(0)
        lastTime = now
        return Outcome("deliver", answer, raw, trace, unverified)
    }

    suspend fun reply(text: String) = run(text, resume = true)

    companion object {
        const val MEMORY_TURNS = 4
        const val ROUND_GAP_MS = 600_000L
        val STOPS = listOf("\nResult:", "\nUser:", "<|endoftext|>")
        val NEW_ROUND = Regex("""^(new (question|topic|round)|start over|forget (that|it|all)|never ?mind)\W*$""", RegexOption.IGNORE_CASE)
        val ANSWER_CUT = Regex("""^(.+?)(?: is the | wrote | directed | composed | developed | is a | does | metres is | is in | plays )""")
    }
}
