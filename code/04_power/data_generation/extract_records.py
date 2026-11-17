"""Extract the power cohort's agent records, programs, reference banks and program benchmark draws.

Reads a copy of the run folder (power-panel-login-20261104 on the cluster, with node-results/,
originals/<case>/workspace/score.py, banks/ and headline_codes.json) and writes compact, scrubbed
files into the package. Session transcripts and archives are not read.

Outputs
  data/agent_runs/power_luna_login/sessions.jsonl.gz   every session attempt (originals and replays)
  data/agent_runs/power_luna_login/originals.csv       the 200 originals with their frozen lead codes
  data/agent_runs/power_luna_login/programs/<session_id>.py   each original's saved score.py
  data/intermediate/04_power/bank_p.csv.gz             each original's program p-value on its reference subsamples
  data/intermediate/04_power/program_draws.csv.gz      each program's p-value on 4,000 new datasets with its own
                                                       effect and on 4,000 with none
Usage: python code/04_power/data_generation/extract_records.py RUN_COPY
"""
import csv
import glob
import gzip
import io
import json
import sys
from pathlib import Path

import generate

ROOT = Path(__file__).resolve().parents[3]
RUN = Path(sys.argv[1])
COHORT = ROOT / "data" / "agent_runs" / "power_luna_login"
INTER = ROOT / "data" / "intermediate" / "04_power"
LOGIN = {"subscription:second": "subscription-A", "subscription:primary": "subscription-B"}
PHASE_KEYS = ("seconds", "returncode", "timed_out", "ok", "error", "n_events", "n_commands", "usage")
(COHORT / "programs").mkdir(parents=True, exist_ok=True)
INTER.mkdir(parents=True, exist_ok=True)


def attempt(r, source):
    """One session attempt, without answer texts, account labels or meter readings."""
    rec = {"session_id": r["case_id"], "role": r["kind"], "record": source, "attempt": r["attempt"],
           "route": LOGIN[r["route"]], "model": "gpt-6-luna", "effort": r["effort"],
           "started": r["started"], "finished": r["finished"], "data_sha256": r["data_sha256"],
           "phases": {p: {k: v[k] for k in PHASE_KEYS if k in v} for p, v in r["phases"].items()},
           "commitment": r.get("commitment"), "has_score_py": r["has_score_py"]}
    if r["kind"] == "original":
        rec.update(slot=r["meta"]["slot"], seed=r["meta"]["seed"], p_value=r.get("u"),
                   program_result=(r.get("original_score") or {}).get("result"),
                   program_returncode=(r.get("original_score") or {}).get("returncode"),
                   final_report=r.get("final_report"))
    else:
        m = r["meta"]
        rec.update(original=m["original"], slot=m["slot"], stratum=m["stratum"], bank_index=m["bank_index"],
                   comparison=r.get("comparison"))
    if source == "rate_limited":
        rec["status"] = "rate_limited"
    elif r["kind"] == "original":
        rec["status"] = "completed" if r.get("u") is not None else "program_error"
    else:
        rec["status"] = "classified" if isinstance(r.get("comparison"), dict) else "cut"
    return rec


records = []
for kind in ("results", "limited"):
    for path in sorted(glob.glob(str(RUN / "node-results" / f"*.{kind}.jsonl"))):
        records += [attempt(json.loads(line), "result" if kind == "results" else "rate_limited")
                    for line in open(path, encoding="utf-8") if line.strip()]
records.sort(key=lambda r: (r["role"] != "original", r["slot"], r["session_id"], r["started"]))
with gzip.GzipFile(COHORT / "sessions.jsonl.gz", "wb", mtime=0) as f:
    for r in records:
        f.write((json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n").encode())

codes = sorted(json.loads((RUN / "headline_codes.json").read_text(encoding="utf-8")), key=lambda c: c["slot"])
with open(COHORT / "originals.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(["session_id", "slot", "seed", "arm", "tau", "lead_outcome", "lead_followup", "lead_family",
                "lead_type"])
    for c in codes:
        w.writerow([c["case_id"], c["slot"], generate.PANEL_SEED + c["slot"], c["arm"], generate.TAU[c["arm"]],
                    c.get("outcome") or "", c.get("followup") or "", c["family"], c.get("type") or ""])
        program = RUN / "originals" / c["case_id"] / "workspace" / "score.py"
        (COHORT / "programs" / f"{c['case_id']}.py").write_bytes(program.read_bytes())

slot_of = {c["case_id"]: c["slot"] for c in codes}
with io.TextIOWrapper(gzip.GzipFile(INTER / "bank_p.csv.gz", "wb", mtime=0), newline="", encoding="utf-8") as g, \
        io.TextIOWrapper(gzip.GzipFile(INTER / "program_draws.csv.gz", "wb", mtime=0), newline="", encoding="utf-8") as h:
    bank, draws = csv.writer(g, lineterminator="\n"), csv.writer(h, lineterminator="\n")
    bank.writerow(["slot", "bank_index", "p_value"])
    draws.writerow(["slot", "effect", "j", "p_value"])
    for cid in sorted(slot_of, key=slot_of.get):
        b = RUN / "banks" / f"{cid}.subsample-raw.json"
        if not b.exists():
            continue
        for j, p in sorted(json.loads(b.read_text(encoding="utf-8"))["scores"].items(), key=lambda kv: int(kv[0])):
            bank.writerow([slot_of[cid], j, "" if p is None else repr(p)])
        fresh = json.loads((RUN / "banks" / f"{cid}.fresh.json").read_text(encoding="utf-8"))["scores"]
        for j, p in sorted(fresh.items(), key=lambda kv: int(kv[0])):
            draws.writerow([slot_of[cid], "yes", j, "" if p is None else repr(p)])
        for j, p in enumerate(json.loads((RUN / "banks" / f"{cid}.null-check.json").read_text(encoding="utf-8"))["scores"]):
            draws.writerow([slot_of[cid], "no", j, "" if p is None else repr(p)])
print(len(records), "session records ->", COHORT.relative_to(ROOT))
