"""Read only audit of what the author actually asks, from local Claude Code
transcripts (and later the web exports). Writes reports/chat_audit.md.
Nothing here enters any training mix; it is a report for a decision.
"""
import collections
import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLAUDE = Path.home() / ".claude" / "projects"
OUT = ROOT / "reports" / "chat_audit.md"

SKIP_PAT = re.compile(r"^\s*(<|/loop|/[a-z-]+\b)|<command-name>|<local-command|\[Request interrupted|"
                      r"Caveat: The messages below|This session is being continued|"
                      r"^\s*(continue|ok|yes|no|go|k|y)\s*[.!]?\s*$", re.I)
GREEK = re.compile(r"[α-ωΑ-Ωά-ώ]")

CATS = {
    "dev, code and bugs": r"\b(fix|bug|error|function|php|python|js|javascript|css|html|deploy|git|api|database|sql|wordpress|plugin|script|code|class|endpoint|query|regex|refactor|test|debug|crash|exception|compile|package|module|dev|build|run|restart)\b",
    "websites and pages": r"\b(site|website|page|landing|layout|responsive|mobile|header|footer|section|menu|navbar|hero|form|checkout|cart|product page|homepage|template|theme|elementor|woocommerce)\b",
    "marketing, seo and ads": r"\b(seo|ads|campaign|google ads|meta ads|facebook|instagram|keyword|copy|conversion|blog|content|ranking|backlink|schema|sitemap|analytics|remarketing|funnel|leads?)\b",
    "design and visuals": r"\b(design|font|colou?r|logo|animation|video|remotion|hyperframes|render|scene|frame|palette|typography|icon|image|thumbnail|3d|three\.?js)\b",
    "agents, loops and automation": r"\b(loop|cron|agent|bot|skill|workflow|automate|automation|scrape|scraper|schedule|scheduled|monitor|watch|pipeline|batch|queue)\b",
    "ai, models and training": r"\b(model|train|training|llm|slm|dataset|embedding|fine ?tune|gpu|tokens?|prompt|inference|benchmark|quantiz|weights|checkpoint)\b",
    "servers, hosting and ops": r"\b(server|ftp|cpanel|dns|ssl|backup|xampp|apache|nginx|domain|hosting|email server|smtp|cron ?job|permissions|windows|powershell|terminal)\b",
    "writing and communication": r"\b(write|email|reply|message|linkedin|post|proposal|invoice|letter|caption|translate|greek|english|tone|rewrite|summari[sz]e)\b",
    "research and explanations": r"\b(what is|what are|how does|how do|explain|educate|why|compare|difference|research|best|recommend|options?|should i|is it better)\b",
    "everyday assistant": r"\b(remind|reminder|timer|alarm|calendar|agenda|meeting|weather|note|shopping|call|schedule a|appointment|tomorrow|tonight)\b",
    "business and clients": r"\b(client|customer|quote|price|pricing|budget|contract|deadline|deliver|project plan|offer|order|shop|store|sales?)\b",
    "system, files and cleanup": r"\b(delete|uninstall|clean|cleanup|disk|folder|files?|move|copy|rename|zip|unzip|download|cache|temp|space|drive|desktop|documents|installed|remove|find)\b",
    "review, audit and checks": r"\b(check|verify|audit|review|pass(?:es)?|confirm|make sure|double check|inspect|compare|scan|validate|proof|look at|status)\b",
    "instructions and rules for you": r"\b(always|never|from now on|rule|remember|do not|don'?t|stop|only|instead|i said|i told|you must|you should|be terse|shorter|no dashes|no emojis)\b",
}
CORRECTION = re.compile(r"\b(no[,.! ]|wrong|not what i|again|why did you|stop|don'?t|dont|never|i said|i told you|you didn'?t|forget|undo|revert|instead)\b", re.I)
FIRST_PERSON = re.compile(r"\b(i am|i'm|i have|i ride|i live|i use|i prefer|i like|i love|i hate|i don'?t like|i work|i dont use|i never|i always|my (?:clients?|business|company|site|car|bike|scooter|wife|kids?|office|phone|laptop|pc|city|home|email|name))\b", re.I)


def human_messages():
    for d in sorted(CLAUDE.iterdir()):
        if not d.is_dir():
            continue
        proj = d.name.replace("C--", "").replace("-", "/")
        for f in d.glob("*.jsonl"):
            try:
                lines = f.open(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for line in lines:
                try:
                    o = json.loads(line)
                except Exception:
                    continue
                if o.get("type") != "user" or o.get("isMeta"):
                    continue
                c = o.get("message", {}).get("content")
                if isinstance(c, list):
                    if any(isinstance(x, dict) and x.get("type") == "tool_result" for x in c):
                        continue
                    c = " ".join(x.get("text", "") for x in c if isinstance(x, dict) and x.get("type") == "text")
                if not isinstance(c, str):
                    continue
                c = c.strip()
                if not c or SKIP_PAT.search(c):
                    continue
                yield proj, c, o.get("timestamp", "")


EXPORTS = ROOT / "corpus" / "exports"


def export_messages():
    """Your turns from the web exports. Assistant turns are never read."""
    f = EXPORTS / "claude" / "human_turns.jsonl"
    if f.exists():
        for line in f.open(encoding="utf-8"):
            o = json.loads(line)
            yield "claude.ai", o["text"].strip(), o.get("at", "")
    f = EXPORTS / "raw" / "chatgpt_user_turns.jsonl"
    if f.exists():
        for line in f.open(encoding="utf-8"):
            try:
                o = json.loads(line)
            except Exception:
                continue
            yield "chatgpt", o["text"].strip(), o.get("at", "")
    f = EXPORTS / "raw" / "gemini_prompts.txt"
    if f.exists():
        for line in f.open(encoding="utf-8"):
            t = line.strip()
            if t.startswith("Prompted "):
                yield "gemini", t[len("Prompted "):].rstrip(".").strip(), ""


def main():
    msgs = [("claude code: " + p, c, ts) for p, c, ts in human_messages()]
    msgs += [(src, c, ts) for src, c, ts in export_messages() if c and not SKIP_PAT.search(c)]
    seen, uniq = set(), []
    for proj, c, ts in msgs:
        key = re.sub(r"\s+", " ", c.lower())[:400]
        if key in seen:
            continue
        seen.add(key)
        uniq.append((proj, c, ts))
    by_proj = collections.Counter(p for p, _, _ in uniq)
    by_src = collections.Counter(p.split(":")[0] for p, _, _ in uniq)
    lengths = [len(c.split()) for _, c, _ in uniq]
    short = sum(1 for l in lengths if l <= 8)
    greek = sum(1 for _, c, _ in uniq if len(GREEK.findall(c)) > 0.3 * max(1, len(re.findall(r"[A-Za-zα-ωΑ-Ω]", c))))
    mixed = sum(1 for _, c, _ in uniq if GREEK.search(c) and re.search(r"[A-Za-z]{3,}", c))
    lower = sum(1 for _, c, _ in uniq if c[:1].islower())
    questions = sum(1 for _, c, _ in uniq if "?" in c)
    pastes = sum(1 for l in lengths if l > 400)
    corrections = [c for _, c, _ in uniq if CORRECTION.search(c) and len(c.split()) <= 40]

    cat_count, cat_ex = collections.Counter(), collections.defaultdict(list)
    for proj, c, _ in uniq:
        low = c.lower()
        scores = {k: len(re.findall(p, low)) for k, p in CATS.items()}
        best = max(scores, key=scores.get)
        if scores[best] == 0:
            best = "other"
        cat_count[best] += 1
        if 4 <= len(c.split()) <= 28 and len(cat_ex[best]) < 8:
            cat_ex[best].append(c.replace("\n", " "))

    facts = collections.Counter()
    for _, c, _ in uniq:
        if len(c.split()) > 120 or c.count("\n") > 8:          # skip pastes and prompt files
            continue
        for sent in re.split(r"(?<=[.!?\n])\s+", c):
            if FIRST_PERSON.search(sent) and 4 <= len(sent.split()) <= 30:
                facts[sent.strip()] += 1
    first_words = collections.Counter(c.split()[0].lower().strip(",.:") for _, c, _ in uniq if c.split())

    n = len(uniq)
    md = [f"# Chat audit, all sources\n",
          f"{len(msgs)} messages of yours after filtering automated prompts, {n} unique. "
          f"Only your turns were read, never the assistant replies. "
          f"Nothing here is in any training mix; this is the evidence for the decision.\n",
          "| source | your messages (unique) |", "|---|---|"]
    md += [f"| {k} | {v} |" for k, v in by_src.most_common()]
    md += ["", "## How you ask\n",
          f"- median length {statistics.median(lengths):.0f} words, mean {statistics.mean(lengths):.0f}; "
          f"{100*short/n:.0f}% are 8 words or fewer; {100*pastes/n:.1f}% are pastes over 400 words",
          f"- {100*greek/n:.1f}% mostly Greek, {100*mixed/n:.1f}% mix Greek and English, the rest English",
          f"- {100*lower/n:.0f}% start lowercase; {100*questions/n:.0f}% contain a question mark",
          f"- {100*len(corrections)/n:.0f}% are corrections or pushback (no, wrong, not what I said, don't, again)",
          "- most common first words: " + ", ".join(f"{w} ({k})" for w, k in first_words.most_common(25)),
          "\n## What you ask for, by primary category\n",
          "| category | messages | share |", "|---|---|---|"]
    for k, v in cat_count.most_common():
        md.append(f"| {k} | {v} | {100*v/n:.1f}% |")
    md.append("\n## Examples per category (short ones, verbatim)\n")
    for k, _ in cat_count.most_common():
        if cat_ex[k]:
            md.append(f"**{k}**")
            md += [f"- {e}" for e in cat_ex[k][:6]]
            md.append("")
    md.append("## Corrections and pushback, what annoys you (sample)\n")
    md += [f"- {c.replace(chr(10), ' ')}" for c in corrections[:25]]
    md.append("\n## Candidate soul facts (first person statements, most repeated first)\n")
    md += [f"- {s.replace(chr(10), ' ')}  (x{k})" for s, k in facts.most_common(45)]
    # ---- proposal: include, exclude, weighting ----
    # client and product names live outside the repo (data/redact_names.txt is
    # gitignored): one lowercase name per line. Missing file means no redaction.
    names_file = ROOT / "data" / "redact_names.txt"
    NAMES = [l.strip().lower() for l in names_file.read_text(encoding="utf-8").splitlines()
             if l.strip()] if names_file.exists() else []
    SENSITIVE = re.compile(r"\b(password|passwd|iban|salary|diagnos|doctor|pain|therapy|divorce|"
                           r"my wife|my kids?|lawsuit|debt|loan|tax return|passport|address is)\b", re.I)
    named = sum(1 for _, c, _ in uniq if any(nm in c.lower() for nm in NAMES))
    sensitive = sum(1 for _, c, _ in uniq if SENSITIVE.search(c))
    WORK_MAP = {
        "dev, code and bugs": "work.code", "websites and pages": "work.code",
        "review, audit and checks": "work.review", "research and explanations": "work.research",
        "ai, models and training": "work.research", "writing and communication": "work.write",
        "marketing, seo and ads": "work.write", "design and visuals": "work.write",
        "system, files and cleanup": "work.files", "servers, hosting and ops": "work.ops",
        "agents, loops and automation": "work.ops", "business and clients": "work.business",
        "instructions and rules for you": "rule.set", "everyday assistant": "everyday (SCENARIOS.md)",
        "other": "unclassified",
    }
    weights = collections.Counter()
    for k, v in cat_count.items():
        weights[WORK_MAP.get(k, "unclassified")] += v
    classified = sum(v for k, v in weights.items() if k != "unclassified")
    md.append("\n## Proposal: what goes in, what stays out\n")
    md.append("Nothing is in yet. Proposed rules, with what they would remove:\n")
    md.append("| rule | messages affected | proposal |")
    md.append("|---|---|---|")
    md.append(f"| client and product names (a name list, redacted before any use) | {named} | phrasing kept, names replaced by placeholders |")
    md.append(f"| personal or sensitive (health, money, family, credentials) | {sensitive} | out entirely |")
    md.append(f"| business and clients category | {cat_count['business and clients']} | gray area: phrasing and intent kept, prices and client details out |")
    md.append(f"| everything else | {n - named - sensitive} | in, as phrasing and frequency evidence only, never as answers |")
    md.append("\n## Proposal: weighting from what you actually ask\n")
    md.append(f"Of the {classified} classified messages, this is the split mapped onto the router's work intents and the everyday list:\n")
    md.append("| target | messages | share of classified | proposed share of trajectories |")
    md.append("|---|---|---|---|")
    everyday_floor = 0.20
    work_total = sum(v for k, v in weights.items() if k not in ("unclassified", "everyday (SCENARIOS.md)"))
    for k, v in weights.most_common():
        if k == "unclassified":
            continue
        share = v / classified
        if k == "everyday (SCENARIOS.md)":
            prop = everyday_floor
        else:
            prop = (1 - everyday_floor) * v / work_total
        md.append(f"| {k} | {v} | {100*share:.1f}% | {100*prop:.0f}% |")
    md.append("")
    md.append(f"The everyday assistant is {100*weights['everyday (SCENARIOS.md)']/classified:.1f}% of what you ask today; "
              "the 20% floor above is a product decision, not a measurement. Everything else follows your real usage.")
    md.append("\n## By project folder\n")
    md.append("| project | your messages |"); md.append("|---|---|")
    md += [f"| {p} | {v} |" for p, v in by_proj.most_common(40)]
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(md), encoding="utf-8", newline="\n")
    print(f"{len(msgs)} messages, {n} unique -> {OUT}")


if __name__ == "__main__":
    main()
