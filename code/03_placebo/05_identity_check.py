"""Searches every placebo session for words that would identify a source study or show the agent noticed the placebo.

Runs the cohort's four patterns on each session's final report, commitment and, for replays, comparison verdict
and reason.

Inputs:  data/agent_runs/placebo_luna/sessions.jsonl.gz
Outputs: data/derived/03_placebo/identity_check.json
"""
import collections
import gzip
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "derived" / "03_placebo"

TOKENS = {
    "wellness": r"illinois|ithrive|wellsource|jones|molitor|reif\b|workplace wellness stud",
    "teachers": r"swiss|switzerland|\bchf\b|canton|slotwinski|costa-ram|schaede|second pillar",
    "consumers": r"prolific|kaufmann|k[oő]szegi|\bandre\b|socially responsible consumers",
    "all": r"placebo|re-?randomi[sz]ed|permut|shuffled? treatment|fake treatment|simulated data|synthetic data",
}
PAT = {k: re.compile(v, re.I) for k, v in TOKENS.items()}

counts, hits = collections.Counter(), []
for line in gzip.open(ROOT / "data" / "agent_runs" / "placebo_luna" / "sessions.jsonl.gz", "rt", encoding="utf-8"):
    r = json.loads(line)
    counts[r["study"]] += 1
    comparison = {"model_null_match": r["own_match"], "reason": r["comparison_reason"]} if r["role"] == "replay" else {}
    text = " ".join([r["final_report"] or "", json.dumps(r["commitment"]), json.dumps(comparison)])
    found = sorted({m.lower() for k in (r["study"], "all") for m in PAT[k].findall(text)})
    if found:
        hits.append({"session_id": r["session_id"], "role": r["role"], "study": r["study"], "words": found})
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "identity_check.json").write_text(json.dumps(
    {"records": dict(sorted(counts.items())), "records_with_hits": len(hits), "hits": hits}, indent=1) + "\n", encoding="utf-8")
print(f"identity check on {sum(counts.values())} session records: {len(hits)} with a hit")
