#!/usr/bin/env python3
"""Extract the persona cohorts' records from the saved run folders. No model calls. Archived as run; not part of run.sh.

Usage: python extract_persona_records.py <research repository root>

Reads, under <root>/output/:
  miao-aligned-pipeline-20261104/runs/<cohort>/<persona>/run_XX/   per run: the analyst's final result and program,
      the inference repair (report, revised program, revised result), the specification summary and three review
      verdicts (stage_runs.py copied the analyst files here; repair_review/r1_repair.mjs and r2_sol.py wrote the rest)
  claude-sonnet-miao-replication-20261027/<arm>/<persona>/run_XX/metadata.json   analyst model, effort, route, date
      and harness version; for GPT-6-Luna on the corrected file the date comes from the Codex thread id (a UUIDv7,
      whose first 48 bits are the start time in milliseconds) in events.jsonl.gz
  miao-luna-persona-pilot-20261026/results_system.jsonl   model, effort and start time for GPT-6-Luna on Miao et al.'s file
Writes, for each cohort, data/agent_runs/persona_<name>/records.csv (one row per anti- or pro-immigration run) and
texts.jsonl.gz (the agents' programs and the repair, summary and review texts). The cohorts' CODEBOOK.md files are
written by hand.
"""
import csv
import gzip
import io
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = Path(sys.argv[1]) / "output"
RUNS = SRC / "miao-aligned-pipeline-20261104" / "runs"
SONNET = SRC / "claude-sonnet-miao-replication-20261027"
LUNA = SRC / "miao-luna-persona-pilot-20261026"

# cohort in the run folders -> (package folder, analyst source arm, data file given to the analyst)
COHORTS = {
    "s46_clean": ("persona_sonnet46_corrected", "results_sonnet-4-6_clean", "corrected"),
    "s55_clean": ("persona_sonnet55_corrected", "results_sonnet-5-5_clean", "corrected"),
    "luna_clean": ("persona_luna_corrected", "results_gpt-6-luna_clean", "corrected"),
    "s46_miao": ("persona_sonnet46_miaofile", "results", "miao"),
    "s55_miao": ("persona_sonnet55_miaofile", "results_sonnet-5-5", "miao"),
    "luna_miao_exact": ("persona_luna_miaofile", None, "miao"),
}
RESULT_FIELDS = ["dep_var", "immigration_measure", "model_type", "controls", "n_obs", "ame", "ame_se", "ame_pvalue"]
# Private folder names in the agents' programs and reports (hard-coded data paths) are replaced by placeholders.
PRIVATE = [(r"/(?:gpfs/accounts|scratch)(?:/[^/\s'\"`…]+){0,2}", "<cluster scratch>"), (r"/gpfs", "<cluster scratch>"),
           (r"/Users/Shared/[^/\s'\"`]+", "<local runtime>"), (r"/Users/[^/\s'\"`]+", "<local folder>"),
           (r"(?:/private)?/var/folders(?:/[^/\s'\"`]+){0,3}", "<temporary folder>")]
# Provenance with no per-run record, from the run plans (claude-sonnet-miao-replication-20261027/README.md,
# miao-luna-persona-pilot-20261026/README.md, miao-aligned-pipeline-20261104/MAIN_PLAN.md and the archived drivers):
CLAUDE_EFFORT = "SDK default"  # run_miao.mjs sets no effort; the Claude Agent SDK's default is high
LUNA_MIAO_FILE = {"analyst_harness": "Codex CLI 0.157", "analyst_route": "subscription"}  # pilot.sbatch, runner.py
REPAIR_MODEL = "claude-sonnet-4-6"  # r1_repair.mjs default; r1_repair.sbatch sets no other model
REVIEW_MODEL = "gpt-6.1-sol"  # r2_sol.py, high effort
ROUTES = {"codex-subscription": "subscription", "subscription": "subscription", "openrouter": "openrouter"}
TEXTS = {"analyst_program": "analysis_final.py", "repair_report": "analysis_final_review.md",
         "repaired_program": "analysis_final_revised.py", "summary": "r2_summary.md",
         "review_1": "r2_verdict_1.txt", "review_2": "r2_verdict_2.txt", "review_3": "r2_verdict_3.txt"}


def scrub(text):
    for pattern, placeholder in PRIVATE:
        text = re.sub(pattern, placeholder, text)
    return text


def final_row(path):
    """The row Miao et al.'s scripts read: model_id == "final", else the last row."""
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    return next((r for r in rows if str(r.get("model_id", "")).strip().lower() == "final"), rows[-1] if rows else {})


def verdict(text):
    """PASS or FAIL from the review's first non-empty line (the fixed format line of the review prompt)."""
    first = next((l.strip().strip("`*") for l in text.splitlines() if l.strip()), "")
    return {"VERDICT: PASS": "PASS", "VERDICT: FAIL": "FAIL"}.get(first.upper(), "")


def analyst_meta(cohort, arm, run):
    if arm is None:  # GPT-6-Luna, Miao et al.'s exact single prompt, persona as the whole system prompt (arm B)
        case_id = (run / "source_case.txt").read_text(encoding="utf-8").split()[0]
        r = next(r for r in map(json.loads, open(LUNA / "results_system.jsonl", encoding="utf-8"))
                 if r["case_id"] == case_id)
        date = datetime.fromtimestamp(r["started"], timezone.utc).date().isoformat()
        return {"session_id": case_id, "analyst_model": r["models_used"][0], "analyst_effort": r["efforts_used"][0],
                **LUNA_MIAO_FILE, "analyst_date": date}
    folder = SONNET / arm / run.parent.name / run.name
    m = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    if m["model"] == "gpt-6-luna":
        with gzip.open(folder / "events.jsonl.gz", "rt", encoding="utf-8") as fh:
            thread = json.loads(fh.readline())["thread_id"]
        start = datetime.fromtimestamp(int(thread.replace("-", "")[:12], 16) / 1000, timezone.utc)
        return {"session_id": f"{cohort}/{run.parent.name}/{run.name}", "analyst_model": m["model"],
                "analyst_effort": m["effort"], "analyst_harness": m["codex_version"].replace("codex-cli", "Codex CLI"),
                "analyst_route": ROUTES[m["route"]], "analyst_date": start.date().isoformat()}
    return {"session_id": f"{cohort}/{run.parent.name}/{run.name}", "analyst_model": m["model"],
            "analyst_effort": CLAUDE_EFFORT,
            "analyst_harness": f"Claude Agent SDK 0.3.197 (Claude Code {m['claude_code_version']})",
            "analyst_route": ROUTES[m.get("route") or "subscription"], "analyst_date": m["timestamp"][:10]}


def write_gzip_jsonl(path, rows):
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
        for r in rows:
            gz.write((json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n").encode())
    path.write_bytes(buf.getvalue())


def main():
    for cohort, (name, arm, data) in COHORTS.items():
        out = ROOT / "data" / "agent_runs" / name
        out.mkdir(parents=True, exist_ok=True)
        records, texts = [], []
        for run in sorted((RUNS / cohort).glob("*_immigration/run_*")):
            a, r = final_row(run / "results_final.csv"), final_row(run / "results_final_revised.csv")
            meta = analyst_meta(cohort, arm, run)
            rec = {"session_id": meta.pop("session_id"), "persona": run.parent.name.split("_")[0], **meta,
                   "data_file": data, "status": "completed"}
            rec.update({f"analyst_{k}": a.get(k, "") for k in RESULT_FIELDS})
            rec.update({f"repaired_{k}": r.get(k, "") for k in RESULT_FIELDS})
            rec.update({"repair_model": REPAIR_MODEL, "review_model": REVIEW_MODEL})
            rec.update({f"review_{k}": verdict((run / f"r2_verdict_{k}.txt").read_text(encoding="utf-8"))
                        for k in (1, 2, 3)})
            records.append(rec)
            texts.append({"session_id": rec["session_id"],
                          **{k: scrub((run / f).read_text(encoding="utf-8")) for k, f in TEXTS.items()}})
        with open(out / "records.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(records[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(records)
        write_gzip_jsonl(out / "texts.jsonl.gz", texts)
        print(name, len(records), "runs")


if __name__ == "__main__":
    main()
