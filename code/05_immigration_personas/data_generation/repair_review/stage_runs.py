#!/usr/bin/env python3
"""Copy each cohort's analyst finals into runs/<cohort>/<persona>/run_XX/ for the aligned pipeline. No model calls.

Only analysis_final.py and results_final.csv are copied (the two files Miao et al.'s repair step reads), so the
replication folders stay untouched. Anti- and pro-immigration runs with a final result only; neutral runs enter no
reported number. Writes runs/cohorts.json: cohort -> source, data file, model, run counts.
Usage: python code/stage_runs.py
"""
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
OUT = HERE.parent
SONNET = OUT / "claude-sonnet-miao-replication-20261027"
LUNA = OUT / "miao-luna-persona-pilot-20261026"
PERSONAS = ("anti_immigration", "pro_immigration")
# (cohort, analyst model, data file, source)
COHORTS = [
    ("s46_miao", "claude-sonnet-4-6", "miao", ("sonnet", "results")),
    ("s46_clean", "claude-sonnet-4-6", "clean", ("sonnet", "results_sonnet-4-6_clean")),
    ("s55_clean", "claude-sonnet-5-5", "clean", ("sonnet", "results_sonnet-5-5_clean")),
    ("luna_clean", "gpt-6-luna", "clean", ("sonnet", "results_gpt-6-luna_clean")),
    ("s55_miao", "claude-sonnet-5-5", "miao", ("sonnet", "results_sonnet-5-5")),
    ("luna_miao_exact", "gpt-6-luna", "miao", ("luna", ["results_system.jsonl"])),
    ("luna_miao_variants", "gpt-6-luna", "miao",
     ("luna", ["results_system.jsonl"] + [f"results_round{k}.jsonl" for k in (2, 3, 4, 5)])),
]


def stage_sonnet(arm, dest):
    n = {}
    for persona in PERSONAS:
        for run in sorted((SONNET / arm / persona).glob("run_*")):
            if not json.loads((run / "metadata.json").read_text()).get("has_final"):
                continue
            d = dest / persona / run.name
            d.mkdir(parents=True, exist_ok=True)
            for f in ("analysis_final.py", "results_final.csv"):
                (d / f).write_bytes((run / f).read_bytes())
            n[persona] = n.get(persona, 0) + 1
    return n


def stage_luna(files, dest):
    # same record selection as issp-application-20261028/code/06_persona_split.py: last non-limited record per case
    records = {}
    for f in files:
        for line in open(LUNA / f):
            r = json.loads(line)
            if not r.get("limited"):
                records[r["case_id"]] = r
    n = {}
    for r in sorted(records.values(), key=lambda r: (r["persona"], r["case_id"])):
        # stance as in 06_persona_split.py: variant personas (brief_anti, advocate_gate_pro, ...) map by name
        st = r.get("stance") or ("anti" if "anti" in r["persona"] else "pro" if "pro" in r["persona"] else "neutral")
        persona, final = f"{st}_immigration", r.get("results_final")
        if persona not in PERSONAS or not (isinstance(final, list) and final and r.get("analysis_final_py")):
            continue
        n[persona] = n.get(persona, 0) + 1
        d = dest / persona / f"run_{n[persona]:02d}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "analysis_final.py").write_text(r["analysis_final_py"])
        cols = list(dict.fromkeys(k for row in final for k in row))
        with open(d / "results_final.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(final)
        (d / "source_case.txt").write_text(f"{r['case_id']} {r['persona']}\n")
    return n


manifest = {}
for cohort, model, data, (kind, src) in COHORTS:
    dest = HERE / "runs" / cohort
    n = stage_sonnet(src, dest) if kind == "sonnet" else stage_luna(src, dest)
    manifest[cohort] = {"analyst_model": model, "data": data, "source": str(src), "runs": n}
    print(cohort, model, data, n)
(HERE / "runs" / "cohorts.json").write_text(json.dumps(manifest, indent=1) + "\n")
