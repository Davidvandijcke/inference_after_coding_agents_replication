"""Build the second blind review: Sol reads every report three ways. No model calls.

opening  the frozen 800-character span from build_review.py, statistics masked, under the revised rule
report   the whole report, statistics masked, under the same rule
facts    the whole report unmasked, for the search facts, which need the numbers

The rule is revised because the first one counted the prompt's own words as hedges (the authors, 30
October). The stance reads see no p-value, and no read sees a model name, slot or the program's
output. One random order, opaque ids, the key in review2/private/. 40 cases of each task are read a
second time for test-retest agreement.

Usage: python3 build_review2.py
"""
import hashlib
import json
import random
import secrets

import build_review as b
import common

ORDER_SEED = 20261032
RETEST = 40
SCOPE = {"opening": "the opening", "report": "the whole text"}


def whole(report, masked):
    if not masked:
        return report.strip()
    lines = [ln for ln in report.splitlines() if not ln.lstrip().startswith("#")]
    return b.mask(" ".join(" ".join(lines).replace("|", " ").replace("*", "").replace("`", "").split()))


def main():
    stance = (common.HERE / "review2_rule_stance.txt").read_text()
    facts = (common.HERE / "review2_rule_facts.txt").read_text()
    kept = common.load_kept()
    rows = []
    for m in b.MODELS:
        src = common.read_jsonl(common.RUN / "luna" / "records.jsonl") if m == "gpt-6-luna" else \
            [r for r in kept.values() if r["model"] == m and r["status"] == "complete"]
        rows += [r for r in src if r.get("final_report")]
    salt = secrets.token_hex(16)
    cases, key = [], {}
    for r in rows:
        report = r["final_report"]
        for task in ("opening", "report", "facts"):
            cid = hashlib.sha256(f"{salt}:{r['model']}:{r['slot']}:{task}".encode()).hexdigest()[:20]
            if task == "facts":
                prompt = facts.replace("{{TEXT}}", whole(report, masked=False))
            else:
                text = b.opening_span(report) if task == "opening" else whole(report, masked=True)
                prompt = stance.replace("{{SCOPE}}", SCOPE[task]).replace("{{TEXT}}", text)
            cases.append({"id": cid, "task": task, "read": 1, "prompt": prompt})
            key[cid] = {"model": r["model"], "slot": r["slot"], "task": task}
    rng = random.Random(ORDER_SEED)
    rng.shuffle(cases)
    for task in ("opening", "report", "facts"):
        cases += [dict(c, read=2) for c in rng.sample([c for c in cases if c["task"] == task], RETEST)]
    rng.shuffle(cases)
    out = common.RUN / "review2"
    (out / "private").mkdir(parents=True, exist_ok=True)
    (out / "private").chmod(0o700)
    with open(out / "cases.jsonl", "w") as f:
        for i, c in enumerate(cases):
            f.write(json.dumps(dict(c, position=i)) + "\n")
    common.write_json(out / "private" / "key.json", key)
    chars = sorted(len(c["prompt"]) for c in cases)
    print(json.dumps({"cases": len(cases), "reports": len(rows), "by_task": {t: sum(c["task"] == t for c in cases) for t in ("opening", "report", "facts")},
                      "prompt_chars_median": chars[len(chars) // 2], "prompt_chars_max": chars[-1],
                      "reports_at_saved_limit": sum(len(r["final_report"]) >= 20000 for r in rows)}))


if __name__ == "__main__":
    main()
