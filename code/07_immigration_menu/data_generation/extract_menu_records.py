#!/usr/bin/env python3
"""Extract the fixed-menu cohorts' records from the saved session records. No model calls. Archived as run.

Usage: python extract_menu_records.py <research repository root>

Reads, under <root>/output/controlled-rebuilt-rerun-20261030/ (anti-immigration analyst) and its pro-arm/
(pro-immigration analyst): roster.json (each replay's workflow, stratum, bank subsample and the rule's own pick),
results/results-*.jsonl and results/rerun-*.jsonl (session records; a rerun's record replaces the first attempt),
pro-arm/results/recovered-selections.jsonl (four selections saved in a subfolder, recovered from the case
archives), runs/original/<workflow>-<k>/ (the six full-data originals per arm) and inputs/ (what every session
received). Writes data/agent_runs/issp_menu_anti/ and issp_menu_pro/: originals.csv, replays.csv,
declarations.jsonl.gz and inputs/.
"""
import csv
import gzip
import io
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = Path(sys.argv[1]) / "output" / "controlled-rebuilt-rerun-20261030"
ARMS = {"issp_menu_anti": SRC, "issp_menu_pro": SRC / "pro-arm"}


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


def as_dict(x):
    return json.loads(x) if isinstance(x, str) else (x or {})


def extract(name, src):
    out = ROOT / "data" / "agent_runs" / name
    out.mkdir(parents=True, exist_ok=True)
    roster = json.loads((src / "roster.json").read_text(encoding="utf-8"))
    final, rerun = {}, set()
    for f in sorted((src / "results").glob("results-*.jsonl")) + sorted((src / "results").glob("rerun-*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                final[r["case_id"]] = r
                if f.name.startswith("rerun-"):
                    rerun.add(r["case_id"])
    recovered = {}
    if (src / "results" / "recovered-selections.jsonl").exists():
        recovered = {x["case_id"]: x["selection"] for x in
                     map(json.loads, (src / "results" / "recovered-selections.jsonl").read_text(encoding="utf-8").splitlines())}
    rows, decl = [], []
    for c in sorted(roster["cases"], key=lambda c: c["position"]):
        r = final.get(c["case_id"], {"status": "not_run"})
        sel = recovered.get(c["case_id"]) or as_dict(r.get("selection"))
        model = sel.get("selected_model_id") if isinstance(sel.get("selected_model_id"), str) else ""
        rows.append({"session_id": c["case_id"], "role": "replay", "position": c["position"],
                     "workflow": c["workflow"], "stratum": c["stratum"], "bank_index": c["bank_index"],
                     "countries": " ".join(c["countries"]), "model": r.get("model", ""),
                     "effort": ",".join(r.get("session_efforts") or []), "route": r.get("route", ""),
                     "status": r["status"], "rerun": int(c["case_id"] in rerun),
                     "selection_recovered_from_archive": int(c["case_id"] in recovered),
                     "selected_model_id": model, "selected_theta": sel.get("theta", ""), "selected_se": sel.get("se", ""),
                     "selected_p": sel.get("pvalue", ""), "rule_pick": c["rule_pick"], "rule_branch": c["rule_branch"] or "",
                     "branch": r.get("branch") or "",
                     "own_match": {True: 1, False: 0}.get((as_dict(r.get("verdict"))).get("model_null_match"), ""),
                     "data_unchanged": {True: 1, False: 0}.get(r.get("data_unchanged"), "")})
        decl.append({"session_id": c["case_id"], "role": "replay", "commitment": as_dict(r.get("commitment")),
                     "comparison": as_dict(r.get("verdict"))})
    originals = []
    for d in sorted((src / "runs" / "original").iterdir()):
        res = json.loads((d / "result.json").read_text(encoding="utf-8"))
        sel = json.loads((d / "selection.json").read_text(encoding="utf-8"))
        originals.append({"session_id": d.name, "role": "original", "workflow": res["workflow"],
                          "replicate": res["replicate"], "model": ",".join(res["session_models"]),
                          "effort": ",".join(res["session_efforts"]), "status": res["status"],
                          "branch": res.get("branch") or "", "selected_model_id": sel["selected_model_id"],
                          "selected_theta": sel["theta"], "selected_se": sel["se"], "selected_p": sel["pvalue"]})
        decl.append({"session_id": d.name, "role": "original", "commitment": as_dict(res.get("commitment"))})
    # Frozen design (prepare_anti.py / prepare_pro.py): each workflow's target analysis and its full-data p-value u.
    if "targets" in roster:
        design = {w: {"target": t, "u": roster["u"]["second" if w == "client_pressure" else "primary"]}
                  for w, t in roster["targets"].items()}
    else:
        design = {w: {"target": roster["target"], "u": roster["u"]}
                  for w in ("fixed_reference", "frontier", "client_pressure")}
    (out / "design.json").write_text(json.dumps(design, indent=1) + "\n", encoding="utf-8")
    write_csv(out / "originals.csv", originals)
    write_csv(out / "replays.csv", rows)
    write_gzip_jsonl(out / "declarations.jsonl.gz", decl)
    # The two arms share every input but the workflow prompts; the shared files are stored once, with the anti arm.
    inputs = out / "inputs"
    inputs.mkdir(exist_ok=True)
    if name == "issp_menu_anti":
        for f in ("ANALYSIS_CONTRACT.md", "CATALOGUE.json", "catalogue_helper.py", "commitment_prompt.txt",
                  "comparison_prompt.txt", "developer-instructions.txt"):
            shutil.copyfile(src / "inputs" / f, inputs / f)
    shutil.copytree(src / "inputs" / "prompts", inputs / "prompts", dirs_exist_ok=True)


if __name__ == "__main__":
    for name, src in ARMS.items():
        extract(name, src)
