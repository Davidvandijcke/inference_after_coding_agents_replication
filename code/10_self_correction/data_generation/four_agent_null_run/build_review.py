"""Build the blind review cases. No model calls.

Every report of every model goes through the same steps: headings and markdown removed, statistics
masked, the first 800 characters kept, cut back to a sentence end. The reviewer sees that text under
an opaque id, in one random order, with no model name, slot or p-value. The key from id to model and
slot stays in review/private/ and is not read until every label is in. The p-value is not read here
at all. 40 cases of each task are read a second time, for test-retest agreement.

Usage: python3 build_review.py [--models M1,M2,...] [--print N]
"""
import hashlib
import json
import random
import re
import secrets
import sys

import common

LIMIT = 800
ORDER_SEED = 20261031
RETEST = 40
MODELS = ("gpt-6-luna", "gpt-6.1-sol", "claude-sonnet-5-5", "claude-opus-5-5")
MASKS = (
    (re.compile(r"\d(?:\.\d+)?\s*[×x]\s*10\s*\^?\S*"), "#"),
    (re.compile(r"\d(?:\.\d+)?e[-−]\d+", re.I), "#"),
    (re.compile(r"[-−+–]?\$?\d[\d,]*\.\d+\s*(?:%|percentage points?|pp\b|points?\b)?"), "#"),
    (re.compile(r"[-−+–]?\$?(?<![\d.])\.\d+"), "#"),
    (re.compile(r"[-−+]?\$\d[\d,]*"), "#"),
    (re.compile(r"[-−+]?\b\d[\d,]*\s*(?:%|percent\b|percentage points?\b|pp\b)"), "#"),
)


def mask(text):
    for pattern, repl in MASKS:
        text = pattern.sub(repl, text)
    return text


def opening_span(report):
    """The fixed span: no headings, no markdown, statistics masked, 800 characters to a sentence end."""
    lines = [ln for ln in (report or "").splitlines() if not re.match(r"\s*#", ln)]
    text = " ".join(" ".join(lines).replace("|", " ").replace("*", "").replace("`", "").split())
    text = mask(text)
    if len(text) <= LIMIT:
        return text
    cut = text[:LIMIT]
    end = max(cut.rfind(". "), cut.rfind(".) "), cut.rfind("? "))
    return cut[:end + 1] if end >= 200 else cut


def main():
    models = list(MODELS)
    if "--models" in sys.argv:
        models = sys.argv[sys.argv.index("--models") + 1].split(",")
    rows = []
    kept = common.load_kept()
    for m in models:
        src = common.read_jsonl(common.RUN / "luna" / "records.jsonl") if m == "gpt-6-luna" else \
            [r for r in kept.values() if r["model"] == m and r["status"] == "complete"]
        rows += [r for r in src if r.get("final_report")]
    if "--print" in sys.argv:
        n = int(sys.argv[sys.argv.index("--print") + 1])
        for r in random.Random(1).sample(rows, n):
            print("-", opening_span(r["final_report"]))
        return
    salt = secrets.token_hex(16)
    tpl = {t: (common.HERE / f"review_rule_{t}.txt").read_text() for t in ("framing", "agreement")}
    cases, key = [], {}
    for r in rows:
        text = opening_span(r["final_report"])
        c = r.get("commitment") or {}
        for task in ("framing", "agreement"):
            if task == "agreement" and not c:
                continue
            cid = hashlib.sha256(f"{salt}:{r['model']}:{r['slot']}:{task}".encode()).hexdigest()[:20]
            prompt = tpl[task].replace("{{OPENING}}", text)
            if task == "agreement":
                prompt = (prompt.replace("{{OUTCOME}}", c["outcome"]).replace("{{POPULATION}}", c["population"])
                          .replace("{{RESTRICTIONS}}", c["sample_restrictions"]).replace("{{NULL}}", c["null"]))
            cases.append({"id": cid, "task": task, "read": 1, "prompt": prompt})
            key[cid] = {"model": r["model"], "slot": r["slot"], "task": task}
    rng = random.Random(ORDER_SEED)
    rng.shuffle(cases)
    for task in ("framing", "agreement"):
        again = [dict(c, read=2) for c in rng.sample([c for c in cases if c["task"] == task], RETEST)]
        cases += again
    rng.shuffle(cases)
    out = common.RUN / "review"
    (out / "private").mkdir(parents=True, exist_ok=True)
    (out / "private").chmod(0o700)
    with open(out / "cases.jsonl", "w") as f:
        for i, c in enumerate(cases):
            f.write(json.dumps(dict(c, position=i)) + "\n")
    common.write_json(out / "private" / "key.json", key)
    chars = [len(c["prompt"]) for c in cases]
    print(json.dumps({"cases": len(cases), "reports": len(rows), "by_task": {t: sum(c["task"] == t for c in cases) for t in ("framing", "agreement")},
                      "prompt_chars_max": max(chars), "prompt_chars_mean": round(sum(chars) / len(chars))}))


if __name__ == "__main__":
    main()
