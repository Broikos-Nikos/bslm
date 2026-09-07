package com.example.llama.bslm

import android.content.Context
import android.content.Intent
import android.media.AudioManager
import android.net.Uri
import android.provider.AlarmClock
import android.provider.MediaStore
import org.json.JSONArray
import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URL
import java.net.URLEncoder
import java.text.Normalizer
import java.text.SimpleDateFormat
import java.util.Calendar
import java.util.Date
import java.util.Locale

/**
 * The model's hands on the phone: the same actions and the same result text
 * as pretrain/agent_tools.py on the PC, so the trained model reads nothing it
 * has not seen the shape of. Network: Wikipedia search and pages, Open-Meteo,
 * YouTube result pages. Local: the clock app for timers and alarms, a small
 * JSON store for reminders, lists, notes and events, the media store for the
 * library, the home state, and the tool registry.
 */
class BslmEnv(private val context: Context, var tools: List<Tool> = emptyList()) {

    data class Tool(val name: String, val desc: String, val cmd: String? = null)

    private val prefs = context.getSharedPreferences("bslm_state", Context.MODE_PRIVATE)
    private var lastResults: List<JSONObject> = emptyList()
    private var lastVideos: List<JSONObject> = emptyList()
    private var lastLibrary: List<Pair<String, Uri?>> = emptyList()
    private var lastList: String? = null
    var playing: String? = null
    private var timerEnd: Long = 0
    private val alarms = mutableListOf<String>()
    private val lights = mutableMapOf<String, String>()
    private val devices = mutableMapOf<String, Boolean>()

    // ------------------------------------------------------------ actions
    fun act(line: String): String {
        val m = Regex("""\s*([a-z_]+)\s*\((.*)\)\s*$""", RegexOption.DOT_MATCHES_ALL).find(line.trim())
            ?: return "unknown action, use one of: search, open, weather, youtube, library, play, timer, alarm, remind, list_add, list_read, note, event, agenda, calc, convert, time, lights, switch, device, volume, tool, ask"
        val name = m.groupValues[1]
        val args = parseArgs(m.groupValues[2])
        fun g(i: Int, d: String = "") = args.getOrNull(i)?.let { if (it is List<*>) it.joinToString(", ") else it.toString() } ?: d
        return try {
            when (name) {
                "search" -> search(g(0))
                "open" -> open(g(0))
                "weather" -> weather(g(0), g(1, "today"))
                "youtube" -> youtube(g(0))
                "library" -> library(g(0))
                "play" -> play(g(0))
                "timer" -> timer(g(0))
                "cancel_timer" -> if (timerEnd > 0) { timerEnd = 0; "Cancelled 1 timer(s)." } else "There is no timer running."
                "timer_left" -> if (timerEnd > System.currentTimeMillis()) "Time left: " + fmtSecs((timerEnd - System.currentTimeMillis()) / 1000) else "No timer is running."
                "alarm" -> alarm(g(0), g(1, "tomorrow"))
                "cancel_alarm" -> if (alarms.isEmpty()) "There is no alarm set." else "Cancelled ${alarms.size} alarm(s).".also { alarms.clear() }
                "remind" -> storeAdd("reminders", JSONObject().put("task", g(0)).put("when", listOf(g(1), g(2)).filter { it.isNotBlank() }.joinToString(" "))).let { "Reminder saved: ${g(0)} (${listOf(g(1), g(2)).filter { it.isNotBlank() }.joinToString(" ")})" }
                "reminders" -> storeList("reminders").let { l -> if (l.isEmpty()) "You have no reminders." else "Reminders:\n" + l.mapIndexed { i, o -> "  ${i + 1}. ${o.getString("task")}  [${o.optString("when")}]" }.joinToString("\n") }
                "list_add" -> listAdd(g(0, "shopping list"), args.getOrNull(1))
                "list_read" -> listRead(g(0, "shopping"))
                "note" -> storeAdd("notes", JSONObject().put("text", g(0))).let { "Noted: ${g(0)}" }
                "notes" -> storeList("notes").let { l -> if (l.isEmpty()) "You have no notes." else "Notes:\n" + l.mapIndexed { i, o -> "  ${i + 1}. ${o.getString("text")}" }.joinToString("\n") }
                "event" -> storeAdd("calendar", JSONObject().put("title", g(0)).put("date", g(1)).put("time", g(2))).let { "Added '${g(0)}' ${g(1)} ${g(2)}".trim() }
                "agenda" -> agenda(g(0, "today"))
                "calc" -> calc(g(0))
                "convert" -> convert(g(0), g(1), g(2))
                "time" -> SimpleDateFormat("'It is' HH:mm 'on' EEEE dd MMMM'.'", Locale.ENGLISH).format(Date())
                "lights" -> lightsAct(g(0, "on"), g(1, "all"), args.getOrNull(2)?.toString())
                "switch" -> switchAct(g(0), g(1, "on"))
                "device" -> switchAct(g(0), g(1, "on")).replace("switched", "turned")
                "volume" -> volume(g(0))
                "tool" -> tool(g(0), g(1, "open"))
                "ask" -> "(waiting for the user)"
                else -> "unknown action $name"
            }
        } catch (e: Exception) {
            "$name failed: ${e.message}"
        }
    }

    private fun parseArgs(inner: String): List<Any> {
        if (inner.isBlank()) return emptyList()
        return try {
            val arr = JSONArray("[$inner]")
            (0 until arr.length()).map { i ->
                val v = arr.get(i)
                if (v is JSONArray) (0 until v.length()).map { v.get(it).toString() } else v
            }
        } catch (e: Exception) {
            listOf(inner.trim().trim('"', '\''))
        }
    }

    // ------------------------------------------------------------ web
    private fun get(url: String, ua: String = UA, accept: String = "*/*"): String {
        val c = URL(url).openConnection() as HttpURLConnection
        c.setRequestProperty("User-Agent", ua)
        c.setRequestProperty("Accept", accept)
        c.setRequestProperty("Accept-Language", "en")
        c.connectTimeout = 15000
        c.readTimeout = 20000
        return c.inputStream.bufferedReader().use { it.readText() }
    }

    private fun clean(s: String) = s.replace(Regex("<[^>]+>"), "").replace("&amp;", "&").replace("&quot;", "\"")
        .replace("&#39;", "'").replace("&lt;", "<").replace("&gt;", ">").replace("\n", " ").trim()

    private fun search(query: String): String {
        val url = "https://en.wikipedia.org/w/api.php?action=query&list=search&format=json&srlimit=5&srsearch=" + enc(query)
        val items = try {
            val arr = JSONObject(get(url)).getJSONObject("query").getJSONArray("search")
            (0 until arr.length()).map { i ->
                val r = arr.getJSONObject(i)
                JSONObject().put("title", r.getString("title")).put("snippet", clean(r.optString("snippet")))
                    .put("url", "https://en.wikipedia.org/wiki/" + r.getString("title").replace(" ", "_"))
            }
        } catch (e: Exception) {
            return "search failed: ${e.message}"
        }
        lastResults = items
        if (items.isEmpty()) return "no results"
        return items.mapIndexed { i, r -> "${i + 1}. ${r.getString("title")}: ${r.getString("snippet").take(220)} (${r.getString("url")})" }.joinToString("\n")
    }

    private fun open(n: String): String {
        val i = n.trim().toIntOrNull() ?: return "open: give a result number"
        if (lastResults.isEmpty()) return "open: there is no list to pick from"
        if (i !in 1..lastResults.size) return "open: there is no result $i, only 1 to ${lastResults.size}"
        val title = lastResults[i - 1].getString("title")
        val page = try {
            get("https://en.m.wikipedia.org/wiki/" + enc(title.replace(" ", "_")).replace("+", "_"), accept = "text/html")
        } catch (e: Exception) {
            return "could not open the page: ${e.message}"
        }
        val rows = mutableListOf<String>()
        Regex("""<table[^>]*class="[^"]*infobox[^"]*"[^>]*>(.*?)</table>""", RegexOption.DOT_MATCHES_ALL).find(page)?.let { box ->
            for (r in Regex("""<th[^>]*>(.*?)</th>\s*<td[^>]*>(.*?)</td>""", RegexOption.DOT_MATCHES_ALL).findAll(box.groupValues[1])) {
                val k = clean(r.groupValues[1].replace(Regex("<br\\s*/?>"), ", "))
                var v = clean(r.groupValues[2].replace(Regex("<(?:br|/li|/p)[^>]*>"), ", "))
                v = v.replace(Regex("\\[\\d+]|\\s*,\\s*,"), "").trim(' ', ',')
                if (k.length in 2..32 && v.length in 1..120 && !Regex("^\\W|website|image|caption|coordinates|native name", RegexOption.IGNORE_CASE).containsMatchIn(k)) rows.add("$k: $v")
                if (rows.size >= 14) break
            }
        }
        val body = (Regex("""<div class="mw-parser-output">(.*)""", RegexOption.DOT_MATCHES_ALL).find(page)?.groupValues?.get(1) ?: page)
            .substringBefore("<h2").replace(Regex("(?s)<table.*?</table>"), " ")
        val lead = Regex("<p[^>]*>(.*?)</p>", RegexOption.DOT_MATCHES_ALL).findAll(body).map { clean(it.groupValues[1]) }.filter { it.isNotBlank() }
            .joinToString(" ").replace(Regex("\\[\\d+]"), "").replace(Regex("\\s+"), " ").trim().take(900)
        return (if (rows.isNotEmpty()) rows.joinToString("\n") + "\n" else "") + lead
    }

    private fun weather(place: String, day: String): String {
        val geo = try {
            JSONObject(get("https://geocoding-api.open-meteo.com/v1/search?count=1&language=en&format=json&name=" + enc(place))).optJSONArray("results")
        } catch (e: Exception) { null }
        if (geo == null || geo.length() == 0) return "no forecast: unknown place '$place'"
        val loc = geo.getJSONObject(0)
        val date = resolveDay(day) ?: return "no forecast: I do not understand the day '$day'"
        val iso = SimpleDateFormat("yyyy-MM-dd", Locale.ENGLISH).format(date.time)
        val h = try {
            JSONObject(get("https://api.open-meteo.com/v1/forecast?hourly=temperature_2m,precipitation_probability&timezone=auto&start_date=$iso&end_date=$iso&latitude=${loc.getDouble("latitude")}&longitude=${loc.getDouble("longitude")}")).getJSONObject("hourly")
        } catch (e: Exception) {
            return "no forecast: ${e.message}"
        }
        val times = h.getJSONArray("time"); val temps = h.getJSONArray("temperature_2m"); val probs = h.getJSONArray("precipitation_probability")
        val parts = mutableListOf<String>(); val rainy = mutableListOf<String>()
        for (i in 0 until times.length()) {
            val t = times.getString(i).substring(11, 16)
            if (t < "06:00") continue
            val p = probs.optInt(i, 0)
            parts.add("$t ${Math.round(temps.optDouble(i, 0.0))}C $p%")
            if (p >= 40) rainy.add(t)
        }
        val summary = if (rainy.isEmpty()) "no rain expected" else "rain likely " + rainy.take(6).joinToString(", ")
        return "${loc.getString("name")}, ${loc.optString("country")}, $iso: ${parts.joinToString(", ")}. $summary."
    }

    private fun youtube(query: String): String {
        val page = try {
            get("https://www.youtube.com/results?search_query=" + enc(query),
                ua = "Mozilla/5.0 (Linux; Android 13) AppleWebKit/537.36 Chrome/128.0 Mobile Safari/537.36", accept = "text/html")
        } catch (e: Exception) {
            return "search failed: ${e.message}"
        }
        val items = mutableListOf<JSONObject>()
        for (m in Regex(""""videoRenderer":\{"videoId":"([^"]+)".*?"title":\{"runs":\[\{"text":"((?:[^"\\]|\\.)*)"""", RegexOption.DOT_MATCHES_ALL).findAll(page)) {
            if (items.size >= 5) break
            items.add(JSONObject().put("id", m.groupValues[1]).put("title", m.groupValues[2].replace("\\\"", "\"").replace("\\u0026", "&")))
        }
        lastVideos = items; lastList = "youtube"
        if (items.isEmpty()) return "no results"
        return items.mapIndexed { i, v -> "${i + 1}. ${v.getString("title")}" }.joinToString("\n")
    }

    private fun library(query: String): String {
        val words = norm(query).split(" ").filter { it.length > 1 }
        val hits = mutableListOf<Pair<String, Uri?>>()
        try {
            val proj = arrayOf(MediaStore.Audio.Media._ID, MediaStore.Audio.Media.TITLE, MediaStore.Audio.Media.ARTIST)
            context.contentResolver.query(MediaStore.Audio.Media.EXTERNAL_CONTENT_URI, proj, null, null, null)?.use { c ->
                while (c.moveToNext() && hits.size < 5) {
                    val s = "${c.getString(1)} - ${c.getString(2)}"
                    if (words.isNotEmpty() && words.all { norm(s).contains(it) }) hits.add(s to Uri.withAppendedPath(MediaStore.Audio.Media.EXTERNAL_CONTENT_URI, c.getLong(0).toString()))
                }
            }
        } catch (e: Exception) { /* no permission: an empty library */ }
        lastLibrary = hits; lastList = "library"
        if (hits.isEmpty()) return "no match in the library"
        return hits.mapIndexed { i, h -> "${i + 1}. ${h.first}" }.joinToString("\n")
    }

    private fun play(n: String): String {
        val i = n.trim().toIntOrNull() ?: return "play: give a result number"
        if (lastList == "youtube") {
            if (lastVideos.isEmpty()) return "play: there is no list to pick from"
            if (i !in 1..lastVideos.size) return "play: there is no result $i, only 1 to ${lastVideos.size}"
            val v = lastVideos[i - 1]
            context.startActivity(Intent(Intent.ACTION_VIEW, Uri.parse("https://www.youtube.com/watch?v=" + v.getString("id"))).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
            playing = v.getString("title")
        } else {
            if (lastLibrary.isEmpty()) return "play: there is no list to pick from"
            if (i !in 1..lastLibrary.size) return "play: there is no result $i, only 1 to ${lastLibrary.size}"
            val h = lastLibrary[i - 1]
            h.second?.let { context.startActivity(Intent(Intent.ACTION_VIEW).setDataAndType(it, "audio/*").addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)) }
            playing = h.first
        }
        return "Playing: $playing"
    }

    // ------------------------------------------------------------ local
    private fun timer(duration: String): String {
        val secs = parseDuration(duration) ?: return "How long should the timer run for?"
        timerEnd = System.currentTimeMillis() + secs * 1000
        try {
            context.startActivity(Intent(AlarmClock.ACTION_SET_TIMER).putExtra(AlarmClock.EXTRA_LENGTH, secs.toInt())
                .putExtra(AlarmClock.EXTRA_MESSAGE, "bslm").putExtra(AlarmClock.EXTRA_SKIP_UI, true).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
        } catch (e: Exception) { /* no clock app: the in app timer still counts */ }
        return "Timer set for ${fmtSecs(secs)}."
    }

    private fun alarm(time: String, date: String): String {
        val (h, mi) = parseClock(time) ?: return "Alarm set for ? ($date)."
        alarms.add("%02d:%02d".format(h, mi))
        try {
            context.startActivity(Intent(AlarmClock.ACTION_SET_ALARM).putExtra(AlarmClock.EXTRA_HOUR, h).putExtra(AlarmClock.EXTRA_MINUTES, mi)
                .putExtra(AlarmClock.EXTRA_MESSAGE, "bslm").putExtra(AlarmClock.EXTRA_SKIP_UI, true).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
        } catch (e: Exception) { }
        return "Alarm set for %02d:%02d (%s).".format(h, mi, date)
    }

    private fun listAdd(name: String, items: Any?): String {
        val ln = name.removeSuffix(" list").trim()
        val add = when (items) { is List<*> -> items.map { it.toString() }; null -> emptyList(); else -> listOf(items.toString()) }.filter { it.isNotBlank() }
        val cur = storeList("list:$ln").map { it.getString("item") }.toMutableList()
        cur.addAll(add)
        storeSet("list:$ln", cur.map { JSONObject().put("item", it) })
        return "Added ${add.joinToString(", ")} to the $ln."
    }

    private fun listRead(name: String): String {
        val ln = name.removeSuffix(" list").trim()
        val cur = storeList("list:$ln").map { it.getString("item") }
        return if (cur.isEmpty()) "The $ln is empty." else "$ln: " + cur.joinToString(", ")
    }

    private fun agenda(day: String): String {
        val evs = storeList("calendar").filter { norm(it.optString("date")) == norm(day) || day == "today" && it.optString("date").isBlank() }
        if (evs.isEmpty()) return "Nothing in the calendar for $day."
        return "Calendar:\n" + evs.joinToString("\n") { "  ${it.optString("time")}  ${it.optString("title")}  ${it.optString("date")}" }
    }

    private fun calc(expr: String): String {
        val e = expr.lowercase().replace("times", "*").replace("x", "*").replace("divided by", "/").replace("plus", "+").replace("minus", "-").replace(",", ".").replace("^", "**")
        val v = try { evalExpr(e) } catch (ex: Exception) { return "I could not compute that." }
        val out = if (v == Math.floor(v) && Math.abs(v) < 1e15) v.toLong().toString() else String.format(Locale.ENGLISH, "%.4f", v).trimEnd('0').trimEnd('.')
        return "$expr = $out"
    }

    private fun evalExpr(s: String): Double = ExprParser(s.replace(" ", "")).expr()

    /** + - * / ** and parentheses, nothing else. */
    private class ExprParser(private val str: String) {
        private var pos = 0
        private fun peek() = if (pos < str.length) str[pos] else ' '
        private fun number(): Double { val st = pos; while (peek().isDigit() || peek() == '.') pos++; return str.substring(st, pos).toDouble() }
        private fun factor(): Double = when {
            peek() == '(' -> { pos++; val v = expr(); pos++; v }
            peek() == '-' -> { pos++; -factor() }
            else -> number()
        }
        private fun power(): Double { var b = factor(); while (str.startsWith("**", pos)) { pos += 2; b = Math.pow(b, factor()) }; return b }
        private fun term(): Double { var v = power(); while (peek() == '*' || peek() == '/') { val op = str[pos++]; val r = power(); v = if (op == '*') v * r else v / r }; return v }
        fun expr(): Double { var v = term(); while (peek() == '+' || peek() == '-') { val op = str[pos++]; val r = term(); v = if (op == '+') v + r else v - r }; return v }
    }

    private fun convert(amount: String, from: String, to: String): String {
        val a = amount.toDoubleOrNull() ?: return "I could not convert that."
        val f = from.lowercase().trimEnd('s'); val t = to.lowercase().trimEnd('s')
        val m = mapOf("km" to 1000.0, "kilometer" to 1000.0, "kilometre" to 1000.0, "mile" to 1609.344, "m" to 1.0, "meter" to 1.0, "metre" to 1.0, "foot" to 0.3048, "feet" to 0.3048, "ft" to 0.3048, "inch" to 0.0254, "cm" to 0.01,
            "kg" to 1.0, "kilo" to 1.0, "kilogram" to 1.0, "pound" to 0.45359237, "lb" to 0.45359237, "g" to 0.001, "gram" to 0.001, "ounce" to 0.0283495, "oz" to 0.0283495,
            "l" to 1.0, "liter" to 1.0, "litre" to 1.0, "gallon" to 3.785411784, "ml" to 0.001)
        if (f in setOf("c", "celsiu") && t in setOf("f", "fahrenheit")) return "$amount $from = ${String.format(Locale.ENGLISH, "%.1f", a * 9 / 5 + 32)} F"
        if (f in setOf("f", "fahrenheit") && t in setOf("c", "celsiu")) return "$amount $from = ${String.format(Locale.ENGLISH, "%.1f", (a - 32) * 5 / 9)} C"
        val mf = m[f] ?: return "I could not convert that."; val mt = m[t] ?: return "I could not convert that."
        return "$amount $from = ${String.format(Locale.ENGLISH, "%.4f", a * mf / mt).trimEnd('0').trimEnd('.')} $to"
    }

    private fun lightsAct(level: String, room: String, color: String?): String {
        val lv = level.lowercase().trim().trimEnd('%')
        val scenes = setOf("movie", "cinema", "reading", "cozy", "relax", "party", "dinner", "focus", "work", "sleep")
        if (lv in scenes) { lights[room] = "$lv scene"; return "Lights in $room: $lv scene" }
        val levels = mapOf("full" to 100, "max" to 100, "bright" to 100, "high" to 80, "on" to 100, "half" to 50, "medium" to 50, "low" to 30, "soft" to 30, "dim" to 30, "night" to 10, "off" to 0)
        val pct = lv.toIntOrNull() ?: levels[lv] ?: return "lights: I do not know the level '$level'; use full, high, half, soft, off, a percentage or a scene"
        lights[room] = if (pct == 0) "off" else "on"
        val extra = listOfNotNull(color?.let { "color $it" }, if (pct in 1..99) (if (lv.all { it.isDigit() }) "brightness $pct%" else "$lv, $pct%") else null)
        return "Lights in $room: ${if (pct == 0) "off" else "on"}" + (if (extra.isNotEmpty()) " (${extra.joinToString(", ")})" else "")
    }

    private fun switchAct(device: String, state: String): String {
        val on = state.lowercase() !in setOf("off", "close", "closed", "stop", "0")
        devices[device] = on
        return "$device switched ${if (on) "on" else "off"}."
    }

    private fun volume(level: String): String {
        val am = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager
        val max = am.getStreamMaxVolume(AudioManager.STREAM_MUSIC)
        val pct = level.trim().toIntOrNull() ?: return "Volume $level."
        am.setStreamVolume(AudioManager.STREAM_MUSIC, (max * pct.coerceIn(0, 100) / 100.0).toInt(), 0)
        return "Volume $pct."
    }

    private fun tool(name: String, what: String): String {
        val want = name.trim().lowercase()
        val t = tools.firstOrNull { want == it.name.lowercase() || want in it.name.lowercase() || it.name.lowercase() in want }
            ?: return "no tool named '$name'; the tools are: " + (tools.joinToString(", ") { it.name }.ifBlank { "none registered" })
        t.cmd?.let { cmd ->
            val launch = context.packageManager.getLaunchIntentForPackage(cmd)
            if (launch != null) context.startActivity(launch.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)) else return "${t.name}: could not run it (no app $cmd)"
        }
        return "${t.name}: $what done"
    }

    // ------------------------------------------------------------ store and helpers
    private fun storeList(key: String): List<JSONObject> {
        val arr = JSONArray(prefs.getString(key, "[]") ?: "[]")
        return (0 until arr.length()).map { arr.getJSONObject(it) }
    }

    private fun storeSet(key: String, items: List<JSONObject>) {
        prefs.edit().putString(key, JSONArray(items).toString()).apply()
    }

    private fun storeAdd(key: String, item: JSONObject) = storeSet(key, storeList(key) + item)

    private fun parseDuration(s: String): Long? {
        var total = 0L
        val words = mapOf("one" to 1, "two" to 2, "three" to 3, "four" to 4, "five" to 5, "six" to 6, "seven" to 7, "eight" to 8, "nine" to 9, "ten" to 10, "fifteen" to 15, "twenty" to 20, "thirty" to 30, "forty" to 40, "fifty" to 50, "sixty" to 60, "half" to 0, "an" to 1, "a" to 1)
        for (m in Regex("""(\d+|[a-z]+)\s*(hour|hr|h|minute|min|m|second|sec|s)\w*""").findAll(s.lowercase())) {
            val n = m.groupValues[1].toLongOrNull() ?: words[m.groupValues[1]]?.toLong() ?: continue
            val unit = m.groupValues[2]
            total += n * when (unit[0]) { 'h' -> 3600; 'm' -> 60; else -> 1 }
        }
        if (s.lowercase().contains("half an hour")) total += 1800
        return if (total > 0) total else null
    }

    private fun fmtSecs(secs: Long): String {
        val h = secs / 3600; val m = (secs % 3600) / 60; val s = secs % 60
        return listOfNotNull(if (h > 0) "${h}h" else null, if (m > 0) "${m}m" else null, if (s > 0 || (h == 0L && m == 0L)) "${s}s" else null).joinToString(" ")
    }

    private fun parseClock(s: String): Pair<Int, Int>? {
        val t = s.lowercase().trim()
        Regex("""(\d{1,2})(?::(\d{2}))?\s*(am|pm)?""").find(t)?.let { m ->
            var h = m.groupValues[1].toInt(); val mi = m.groupValues[2].toIntOrNull() ?: 0
            if (m.groupValues[3] == "pm" && h < 12) h += 12
            if (m.groupValues[3] == "am" && h == 12) h = 0
            if (h in 0..23 && mi in 0..59) return h to mi
        }
        if (t.startsWith("quarter past ")) parseClock(t.removePrefix("quarter past "))?.let { return it.first to 15 }
        if (t.startsWith("half past ")) parseClock(t.removePrefix("half past "))?.let { return it.first to 30 }
        return null
    }

    private fun resolveDay(day: String): Calendar? {
        val c = Calendar.getInstance()
        val d = day.lowercase().trim()
        val days = listOf("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
        when {
            d in setOf("today", "now", "") -> {}
            d == "tomorrow" -> c.add(Calendar.DAY_OF_YEAR, 1)
            d == "the day after tomorrow" || d == "day after tomorrow" -> c.add(Calendar.DAY_OF_YEAR, 2)
            d in days -> { val want = days.indexOf(d); while ((c.get(Calendar.DAY_OF_WEEK) + 5) % 7 != want) c.add(Calendar.DAY_OF_YEAR, 1) }
            Regex("""\d{4}-\d{2}-\d{2}""").matches(d) -> { c.time = SimpleDateFormat("yyyy-MM-dd", Locale.ENGLISH).parse(d) ?: return null }
            else -> return null
        }
        return c
    }

    companion object {
        const val UA = "bslm-android/0.1 (small language model experiment; github.com/Broikos-Nikos/bslm)"
        fun enc(s: String): String = URLEncoder.encode(s, "UTF-8")
        fun norm(s: String): String = Normalizer.normalize(s.lowercase(), Normalizer.Form.NFD)
            .replace(Regex("\\p{M}"), "").replace(Regex("[^a-z0-9 ]+"), " ").trim()
    }
}
