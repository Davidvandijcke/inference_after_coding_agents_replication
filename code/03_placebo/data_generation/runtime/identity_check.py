"""Whether any session named its source study (no model calls, report only).

Searches every kept record's final report, commitment and comparison reason for words that
would identify a source paper or its setting. A hit is reported with its slot and stratum;
nothing is excluded, relabeled or rerun because of it.

Usage: python identity_check.py OUT.json
"""
import collections
import json
import re
import sys

import runner

TOKENS = {
    "wellness": r"illinois|ithrive|wellsource|jones|molitor|reif\b|workplace wellness stud",
    "teachers": r"swiss|switzerland|\bchf\b|canton|slotwinski|costa-ram|schaede|second pillar",
    "consumers": r"prolific|kaufmann|k[oő]szegi|\bandre\b|socially responsible consumers",
    "all": r"placebo|re-?randomi[sz]ed|permut|shuffled? treatment|fake treatment|simulated data|synthetic data",
}
PAT = {k: re.compile(v, re.I) for k, v in TOKENS.items()}

hits = []
n = collections.Counter()
for r in runner.load_results().values():
    exp = r.get("meta", {}).get("experiment")
    if exp is None:
        continue
    n[exp] += 1
    text = " ".join([r.get("final_report") or "", json.dumps(r.get("commitment") or {}),
                     json.dumps(r.get("comparison") or {})])
    found = sorted({m.lower() for k in (exp, "all") for m in PAT[k].findall(text)})
    if found:
        hits.append({"case_id": r["case_id"], "kind": r.get("kind"), "experiment": exp, "slot": r["meta"].get("slot"),
                     "stratum": r["meta"].get("stratum"), "words": found})
out = {"records": dict(n), "records_with_hits": len(hits), "by_word": dict(collections.Counter(w for h in hits for w in h["words"])),
       "hits": hits}
runner.write_json(sys.argv[1], out)
print(json.dumps({k: v for k, v in out.items() if k != "hits"}))
