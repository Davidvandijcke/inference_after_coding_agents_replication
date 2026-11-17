"""Export the placebo cohort's session records, programs and reference banks from the cluster run folder.

Run once by the authors on local copies of the run folder's node-results/ and banks/ and of the six node
archives of the original sessions (no model calls). Writes:
  data/agent_runs/placebo_luna/sessions.jsonl.gz   one record per session (102 originals, 24,440 replays)
  data/agent_runs/placebo_luna/programs/           each original's frozen program, score_slotNNN.py
  data/intermediate/03_placebo/bank_pvalues.csv.gz each original's reference bank: the frozen program's
                                                   p-value on every scored subsample

A case's record is chosen as the runner chose it (runner.load_results): files in name order, a classified
record never displaced by an unclassified one. Attempts the provider cut ("limited" records) are counted
per case, not shipped. Dropped fields: the agents' raw phase answers (repeated in final_report, commitment
and comparison), the CLI's internal session ids and the account usage meter. The wellness extension of
3 November (campaign main-x3, cancelled before any replay) is not part of the cohort and is skipped.

Usage: python export_cluster_records.py RUN_COPY ARCHIVE_COPY PACKAGE_ROOT
"""
import collections
import csv
import glob
import gzip
import io
import json
import sys
import tarfile
from pathlib import Path

run, archives, package = map(Path, sys.argv[1:4])
out = package / "data" / "agent_runs" / "placebo_luna"
STUDIES = ("wellness", "teachers", "consumers")
FIELDS = ("model", "null", "outcome", "population", "sample_restrictions")
cohort = [json.loads(l) for l in open(run / "all_originals.jsonl", encoding="utf-8")]
originals = {c["case_id"]: c["meta"]["slot"] for c in cohort}
assert sorted(originals.values()) == list(range(102))

records, cuts = {}, collections.Counter()
for path in sorted(glob.glob(str(run / "node-results" / "*.jsonl"))):
    if "/main-x3-" in path:
        continue
    for line in open(path, encoding="utf-8"):
        rec = json.loads(line)
        if path.endswith(".limited.jsonl"):
            cuts[rec["case_id"]] += 1
            continue
        prev = records.get(rec["case_id"])
        if prev is None or isinstance(rec.get("comparison"), dict) or not isinstance(prev.get("comparison"), dict):
            records[rec["case_id"]] = rec


def session(rec):
    m = rec["meta"]
    original = rec["kind"] == "original"
    slot = m["slot"]
    assert m["experiment"] == STUDIES[slot % 3]
    assert (originals[rec["case_id"]] if original else originals[m["original"]]) == slot
    row = {"session_id": rec["case_id"], "role": rec["kind"], "cohort": "placebo_luna", "slot": slot,
           "study": m["experiment"], "model": "gpt-6-luna", "effort": rec["effort"], "harness": "codex-cli 0.157.0",
           "route": rec["route"], "attempt": rec["attempt"], "attempts_cut": cuts[rec["case_id"]],
           "started": rec["started"], "finished": rec["finished"], "data_sha256": rec["data_sha256"]}
    if original:
        row.update(seed=m["seed"], n=m["n"], has_score_py=rec["has_score_py"], p_value=rec["u"],
                   score={k: rec["original_score"]["result"][k] for k in ("estimate", "std_error", "p_value")})
    else:
        row.update(original_session_id=m["original"], stratum=m["stratum"], bin=m["bin"], bank_index=m["bank_index"],
                   bank_p_value=m["ref_p"], own_match=rec["comparison"]["model_null_match"],
                   comparison_reason=rec["comparison"]["reason"])
    row["status"] = "completed" if original else "classified"
    row["phases"] = {name: {k: p[k] for k in ("seconds", "returncode", "timed_out", "ok", "n_events", "n_commands", "usage")}
                     for name, p in rec["phases"].items()}
    row["commitment"] = {k: (rec["commitment"] or {}).get(k) for k in FIELDS}
    row["final_report"] = rec["final_report"]
    return row


rows = sorted((session(r) for r in records.values()), key=lambda r: (r["slot"], r["role"] != "original", r["session_id"]))
assert collections.Counter(r["role"] for r in rows) == {"original": 102, "replay": 24440}
with gzip.GzipFile(out / "sessions.jsonl.gz", "w", mtime=0) as gz:
    for r in rows:
        gz.write((json.dumps(r, ensure_ascii=False) + "\n").encode())

# Frozen programs: workspace/score.py of each original's archived case folder.
(out / "programs").mkdir(exist_ok=True)
attempt = {r["session_id"]: r["attempt"] for r in rows if r["role"] == "original"}
found = 0
for node in sorted(archives.glob("*.tar")):
    with tarfile.open(node) as t:
        for member in t.getmembers():
            cid, _, a = member.name.removesuffix(".tar.gz").partition("-a")
            if cid in attempt and int(a) == attempt[cid]:
                with tarfile.open(fileobj=io.BytesIO(t.extractfile(member).read()), mode="r:gz") as case:
                    program = case.extractfile(f"{cid}/workspace/score.py").read()
                (out / "programs" / f"score_slot{originals[cid]:03d}.py").write_bytes(program)
                found += 1
assert found == 102, found

# Reference banks: each subsample's index and the frozen program's p-value on it (empty when the program
# failed on that subsample; such a subsample is in neither stratum).
bank_out = package / "data" / "intermediate" / "03_placebo"
with gzip.GzipFile(bank_out / "bank_pvalues.csv.gz", "w", mtime=0) as gz:
    text = io.TextIOWrapper(gz, newline="", encoding="utf-8")
    w = csv.writer(text, lineterminator="\n")
    w.writerow(["slot", "bank_index", "p_value"])
    for cid, slot in sorted(originals.items(), key=lambda kv: kv[1]):
        bank = json.load(open(run / "banks" / f"{cid}.subsample-raw.json", encoding="utf-8"))
        assert bank["slot"] == slot and bank["m"] == int(0.10 * bank["n"] + 0.5)
        for j, p in sorted(((int(j), p) for j, p in bank["scores"].items())):
            w.writerow([slot, j, "" if p is None else repr(p)])
    text.flush()
    text.detach()
print("exported", len(rows), "sessions,", found, "programs")
