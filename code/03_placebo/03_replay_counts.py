"""Tail and body strata of each placebo original, and its replays and own matches in each.

A bank subsample is in the tail when the original's program gives p at most u on it. Checks each replay's recorded
stratum and bank p-value against the bank.

Inputs:  data/agent_runs/placebo_luna/sessions.jsonl.gz, data/intermediate/03_placebo/bank_pvalues.csv.gz
Outputs: data/derived/03_placebo/replay_counts.csv
"""
import collections
import csv
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "derived" / "03_placebo"

sessions = [json.loads(l) for l in gzip.open(ROOT / "data" / "agent_runs" / "placebo_luna" / "sessions.jsonl.gz", "rt", encoding="utf-8")]
originals = {r["slot"]: r for r in sessions if r["role"] == "original"}
bank = collections.defaultdict(dict)
for r in csv.DictReader(gzip.open(ROOT / "data" / "intermediate" / "03_placebo" / "bank_pvalues.csv.gz", "rt", encoding="utf-8")):
    if r["p_value"]:  # empty: the program failed on this subsample, which is in neither stratum
        bank[int(r["slot"])][int(r["bank_index"])] = float(r["p_value"])

counts = {s: collections.Counter() for s in originals}
for r in sessions:
    if r["role"] != "replay":
        continue
    s, u = r["slot"], originals[r["slot"]]["p_value"]
    p = bank[s].get(r["bank_index"])
    if p is None or p != r["bank_p_value"] or (p <= u) != (r["stratum"] == "tail"):
        sys.exit(f"replay {r['session_id']} (slot {s}, subsample {r['bank_index']}): its recorded stratum or bank "
                 "p-value disagrees with the bank")
    if r["status"] != "classified":
        sys.exit(f"replay {r['session_id']}: status {r['status']!r}, expected 'classified'")
    counts[s][r["stratum"]] += 1
    counts[s][r["stratum"] + "_matches"] += r["own_match"]

OUT.mkdir(parents=True, exist_ok=True)
with open(OUT / "replay_counts.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(["slot", "study", "u", "bank_tail", "bank_body", "tail_replays", "tail_matches", "body_replays", "body_matches"])
    for s in sorted(originals):
        u, c = originals[s]["p_value"], counts[s]
        tail = sum(p <= u for p in bank[s].values())
        w.writerow([s, originals[s]["study"], repr(u), tail, len(bank[s]) - tail,
                    c["tail"], c["tail_matches"], c["body"], c["body_matches"]])
print(f"{sum(sum(c[k] for k in ('tail', 'body')) for c in counts.values())} replays counted for {len(originals)} originals")
