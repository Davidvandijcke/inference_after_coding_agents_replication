"""Join the Sol Card blind reads to their case ids. No model calls. Writes results/sol_card_facts.csv.

Usage: sbatch analysis.sbatch export_card_review.py
"""
import csv
import glob
import json

import common

R = common.RUN
key = json.loads((R / "review7" / "private" / "key.json").read_text())
FIELDS = ("case_id", "read", "nominal_p", "tests_stated", "other_significant", "own_correction", "correction_tests", "adjusted_p", "survives", "quote")
rows = []
for p in sorted(glob.glob(str(R / "review7" / "out" / "results-*.jsonl"))):
    for r in common.read_jsonl(p):
        if r.get("label"):
            rows.append({"case_id": key[r["id"]]["case_id"], "read": r["read"], **{f: r["label"].get(f) for f in FIELDS[2:]}})
rows.sort(key=lambda x: (x["case_id"], x["read"]))
with open(R / "results" / "sol_card_facts.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=FIELDS)
    w.writeheader()
    w.writerows(rows)
print(len(rows), "labels of", len(key), "reports plus repeats")
