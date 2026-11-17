#!/usr/bin/env python3
"""Extract the realistic-request cohorts' records from the saved session records. No model calls. Archived as run.

Usage: python extract_request_records.py <research repository root>

Reads, under <root>/output/:
  issp-luna-selection-20261027/   originals.jsonl and node-results/*.jsonl (the three full-data originals and the
                                  1,000 replays; files named issp-rep2-* hold the rerun job's records),
                                  banks/<original>.replays.jsonl (each replay's subsample and stratum),
                                  review/reviewed_verdicts.json and review/wording_candidates.jsonl (the second reading
                                  of the replays' verdicts), source/ (the prompts and codebook every session received)
  issp-application-20261028/original/emitted-analysis_final.py   the original's program, copied from its workspace
  issp-mvalue-reference-20261029/ cases.jsonl, node-results/*.jsonl and source/ (the 60 full-data reference sessions
                                  and their two prompt variants)
Writes:
  data/agent_runs/issp_request_luna/  originals.csv, replays.csv, declarations.jsonl.gz, review_candidates.csv,
                                      inputs/, original/ (the selected original's analysis log, brief, result,
                                      closing message and program)
  data/agent_runs/issp_mvalue_luna/   sessions.csv, analysis_paths.jsonl.gz, inputs/
The original's analysis log names the cluster's Python interpreter in four commands; that path is replaced by
"<cluster python>".
"""
import csv
import glob
import gzip
import io
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = Path(sys.argv[1]) / "output"
RUN = SRC / "issp-luna-selection-20261027"
REF = SRC / "issp-mvalue-reference-20261029"
ORIGINAL = "7737e9064880f41217902b7cce1562f6"
# The second reading of the verdicts (review/build_reviewed.py, "Reviewed secondary tally (Claude, 28 October 2026)";
# issp-application-20261028/README.md, "Matching verdicts") was done by a Claude Code session running Claude Opus 5.5.
REVIEW_READER = "claude-opus-5.5 (Claude Code session)"
# The interpreter's absolute path in the cluster's software tree (the only absolute .../bin/python path in the log).
CLUSTER_PYTHON = re.compile(r"/[\w.+-]+(?:/[\w.+-]+)*/bin/python")
OUT = ROOT / "data" / "agent_runs" / "issp_request_luna"
OUT_REF = ROOT / "data" / "agent_runs" / "issp_mvalue_luna"


def records(folder, kind="results"):
    """Records by case id; a later file's record replaces an earlier one. Also returns each case's file name."""
    out, source = {}, {}
    for p in sorted(glob.glob(str(folder / "node-results" / f"*.{kind}.jsonl"))):
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            out[r["case_id"]], source[r["case_id"]] = r, Path(p).name
    return out, source


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def write_gzip_jsonl(path, rows):
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
        for r in rows:
            gz.write((json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n").encode())
    path.write_bytes(buf.getvalue())


def status(r):
    return "completed" if all(ph.get("ok") for ph in r["phases"].values()) else "failed"


def request_cohort():
    res, source = records(RUN)
    limited = set(records(RUN, "limited")[0])
    originals = sorted((json.loads(l) for l in open(RUN / "originals.jsonl", encoding="utf-8")),
                       key=lambda c: c["meta"]["slot"])
    cases = [json.loads(l) for l in open(RUN / "banks" / f"{ORIGINAL}.replays.jsonl", encoding="utf-8")]
    reviewed = json.loads((RUN / "review" / "reviewed_verdicts.json").read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "originals.csv", [
        {"session_id": c["case_id"], "role": "original", "slot": c["meta"]["slot"], "model": "gpt-6-luna",
         "effort": res[c["case_id"]]["effort"], "route": res[c["case_id"]]["route"], "status": status(res[c["case_id"]]),
         "data_sha256": res[c["case_id"]]["data_sha256"], "program_p": repr(res[c["case_id"]]["u"]),
         "selected_as_original": int(c["case_id"] == ORIGINAL)} for c in originals])
    rows, decl = [], []
    for c in sorted(cases, key=lambda c: c["meta"]["bank_index"]):
        r = res[c["case_id"]]
        own = r["comparison"]["model_null_match"]
        v = reviewed.get(c["case_id"])
        assert isinstance(own, bool) and (v is None or v["raw"] is False)
        rows.append({"session_id": c["case_id"], "role": "replay", "model": "gpt-6-luna", "effort": r["effort"],
                     "route": r["route"], "stratum": c["meta"]["stratum"], "bank_index": c["meta"]["bank_index"],
                     "subsample_p": repr(c["meta"]["subsample_p"]), "data_sha256": r["data_sha256"],
                     "status": status(r), "retried_after_usage_limit": int(c["case_id"] in limited),
                     "completed_in_rerun_job": int(source[c["case_id"]].startswith("issp-rep2-")),
                     "own_match": int(own),
                     "reviewed_match": int(own or (v is not None and v["reviewed"] is True)),
                     "review_category": "+".join(v["categories"]) if v else "",
                     "review_uncertain": int(v is not None and v["reviewed"] == "uncertain"),
                     "review_reader": REVIEW_READER})
        decl.append({"session_id": c["case_id"], "role": "replay", "commitment": r["commitment"],
                     "comparison": r["comparison"]})
    for c in originals:
        decl.append({"session_id": c["case_id"], "role": "original", "commitment": res[c["case_id"]]["commitment"]})
    write_csv(OUT / "replays.csv", rows)
    write_gzip_jsonl(OUT / "declarations.jsonl.gz", decl)
    candidates = [json.loads(l) for l in open(RUN / "review" / "wording_candidates.jsonl", encoding="utf-8")]
    write_csv(OUT / "review_candidates.csv", [{"i": c["i"], "session_id": c["case_id"], "stratum": c["stratum"]}
                                              for c in sorted(candidates, key=lambda c: int(c["i"]))])
    o = res[ORIGINAL]
    (OUT / "original" / "programs").mkdir(parents=True, exist_ok=True)
    (OUT / "original" / "ANALYSIS_PATHS.jsonl").write_text(
        CLUSTER_PYTHON.sub("<cluster python>", o["ANALYSIS_PATHS.jsonl"]), encoding="utf-8")
    (OUT / "original" / "BRIEF.md").write_text(o["BRIEF.md"], encoding="utf-8")
    (OUT / "original" / "results_final.json").write_text(o["results_final.json"], encoding="utf-8")
    (OUT / "original" / "final_report.md").write_text(o["final_report"], encoding="utf-8")
    shutil.copyfile(SRC / "issp-application-20261028" / "original" / "emitted-analysis_final.py",
                    OUT / "original" / "programs" / "analysis_final.py")
    (OUT / "inputs").mkdir(exist_ok=True)
    for name in ("research_prompt.txt", "CODEBOOK.md", "commitment_prompt.txt", "comparison_prompt.txt"):
        shutil.copyfile(RUN / "source" / name, OUT / "inputs" / name)


def reference_cohort():
    res = records(REF)[0]
    cases = [json.loads(l) for l in open(REF / "cases.jsonl", encoding="utf-8")]
    OUT_REF.mkdir(parents=True, exist_ok=True)
    rows, logs = [], []
    for c in sorted(cases, key=lambda c: (c["meta"]["brief"], c["meta"]["session"])):
        r = res[c["case_id"]]
        rows.append({"session_id": c["case_id"], "role": "reference", "brief": c["meta"]["brief"],
                     "session": c["meta"]["session"], "prompt_file": c["prompt_file"], "model": "gpt-6-luna",
                     "effort": r["effort"], "route": r["route"], "status": status(r), "data_sha256": r["data_sha256"]})
        logs.append({"session_id": c["case_id"], "analysis_paths": r.get("ANALYSIS_PATHS.jsonl") or "",
                     "results_final": r.get("results_final.json") or ""})
    write_csv(OUT_REF / "sessions.csv", rows)
    write_gzip_jsonl(OUT_REF / "analysis_paths.jsonl.gz", logs)
    (OUT_REF / "inputs").mkdir(exist_ok=True)
    for name in ("research_prompt_anti.txt", "research_prompt_nonpartisan.txt"):
        shutil.copyfile(REF / "source" / name, OUT_REF / "inputs" / name)


if __name__ == "__main__":
    request_cohort()
    reference_cohort()
