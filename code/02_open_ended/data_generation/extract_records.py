"""Extract the open-ended cohort's records into the package.

Run once by the authors (no model calls); kept so the provenance of every shipped record is
explicit. Inputs:
  NODE_RESULTS  a copy of the run's node-results/ folder from the cluster (files main-*.jsonl):
                every session record the runner wrote, one JSON line per attempt, in the node's
                results file or, for attempts stopped by a usage limit, its limited file
  RESEARCH      the research repository: the run folder output/open-ended-sim-redesign-20261026/
                (launch manifest, served datasets, reference banks, the run's per-original counts)
                and the archived original workspaces in
                output/multi-model-originals-20261030/archive/originals-gpt-6-luna.tar.gz

Writes
  data/raw/pathways/main-NN.csv, CODEBOOK.md     the 100 datasets and the codebook as served
  data/agent_runs/open_ended_luna/originals.csv  one row per original: u, estimate, commitment
  data/agent_runs/open_ended_luna/sessions.jsonl.gz  every session attempt, originals and replays,
      without the agents' answer texts
  data/agent_runs/open_ended_luna/reports.jsonl.gz   each original's final report
  data/agent_runs/open_ended_luna/programs/slot-NN/  each original's workspace (score.py)
  data/intermediate/02_open_ended/bank_pvalues.csv.gz  each original program's p-value on
      every subsample of its reference bank

Usage: python extract_records.py RESEARCH NODE_RESULTS
"""
import csv
import gzip
import hashlib
import io
import json
import re
import shutil
import sys
import tarfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUN = Path(sys.argv[1]) / "output" / "open-ended-sim-redesign-20261026"
ARCHIVE = Path(sys.argv[1]) / "output" / "multi-model-originals-20261030" / "archive" / "originals-gpt-6-luna.tar.gz"
NODE = Path(sys.argv[2])
MAIN = RUN / "two-arm" / "main"
COHORT = ROOT / "data" / "agent_runs" / "open_ended_luna"
RAW = ROOT / "data" / "raw" / "pathways"
INTER = ROOT / "data" / "intermediate" / "02_open_ended"
FIELDS = ("model", "null", "population", "outcome", "sample_restrictions")
COMPONENT = {"subsample-raw": "base", "subsample-raw-ext1": "ext1"}
PRIVATE = re.compile(r"/scratch/|/gpfs/|/home/|/Users/|@[\w.-]+\.(edu|com|org)")  # cluster paths, emails


def g17(x):
    return "" if x is None else repr(float(x))


for d in (COHORT / "programs", RAW, INTER):
    d.mkdir(parents=True, exist_ok=True)

# Datasets and codebook, checked against the hashes frozen before launch.
manifest = json.loads((RUN / "main" / "MANIFEST.json").read_text(encoding="utf-8"))
for name, sha in manifest["original_datasets"].items():
    src = RUN / "main" / "stage" / "data" / "originals" / name
    assert hashlib.sha256(src.read_bytes()).hexdigest() == sha, name
    shutil.copyfile(src, RAW / name)
codebook = RUN / "pilot" / "source" / "CODEBOOK.md"
assert hashlib.sha256(codebook.read_bytes()).hexdigest() == manifest["source_files"]["CODEBOOK.md"]
shutil.copyfile(codebook, RAW / "CODEBOOK.md")

# Session records, in the order the runner's analysis read them (file name, then line).
records = []
for p in sorted(NODE.glob("main-*.jsonl")):
    stem, sink = p.name.split(".")[0], p.name.split(".")[1]
    for i, line in enumerate(open(p, encoding="utf-8")):
        records.append((stem, sink, i, json.loads(line)))
originals = {r["meta"]["slot"]: r for _, _, _, r in records if r["kind"] == "original"}
assert sorted(originals) == list(range(100))
seeds = {json.loads(l)["meta"]["slot"]: json.loads(l)["meta"]["seed"]
         for l in open(RUN / "main" / "stage" / "main_originals.jsonl", encoding="utf-8") if l.strip()}
pooled = {r["slot"]: r for r in json.loads((MAIN / "ext1_pooled_subsample_raw.json").read_text(encoding="utf-8"))["rows"]}


def phase(ph):
    keep = ("seconds", "returncode", "timed_out", "ok", "error", "limit_error", "n_events", "n_commands", "usage")
    return {k: ph[k] for k in keep if k in ph}


def session(stem, sink, line, r):
    m = r["meta"]
    out = {"session_id": r["case_id"], "role": r["kind"], "slot": m["slot"]}
    if r["kind"] == "replay":
        out.update(component=COMPONENT[m["arm"]], stratum=m["stratum"], bank_index=m["bank_index"], ref_p=m["ref_p"])
    out.update(file=stem, sink=sink, line=line, attempt=r["attempt"], route=r["route"], effort=r["effort"],
               started=r["started"], finished=r["finished"], data_sha256=r["data_sha256"],
               has_score_py=r["has_score_py"], commitment=r.get("commitment"))
    if r["kind"] == "replay":
        cmp_ = r.get("comparison")
        out.update(model_null_match=cmp_["model_null_match"] if isinstance(cmp_, dict) else None,
                   reason=cmp_["reason"] if isinstance(cmp_, dict) else None)
    else:
        s = r["original_score"]
        out.update(u=r["u"], score_returncode=s["returncode"], score_result=s["result"])
    out["phases"] = {k: phase(v) for k, v in r["phases"].items()}
    assert not PRIVATE.search(json.dumps(out)), r["case_id"]
    return out


sessions = [session(*x) for x in records]
roles = Counter((s["role"], s.get("component"), s["sink"]) for s in sessions)
print(dict(roles))

# Originals: the records agree with the run's compact copies; slot 27's program fails.
with open(COHORT / "originals.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(["slot", "seed", "dataset", "original_id", "model", "effort", "harness", "route", "status",
                "usable", "u", "estimate", "std_error", *[f"commitment_{x}" for x in FIELDS]])
    for s in range(100):
        r = originals[s]
        res = r["original_score"]["result"]
        assert (s in pooled) == (r["u"] is not None) and (s not in pooled or pooled[s]["u"] == r["u"])
        assert seeds[s] == r["meta"]["seed"]
        w.writerow([s, r["meta"]["seed"], f"main-{s:02d}.csv", r["case_id"], "gpt-6-luna", r["effort"],
                    "codex-cli 0.157.0", r["route"], "completed", "TRUE" if r["u"] is not None else "FALSE",
                    g17(r["u"]), g17(res and res["estimate"]), g17(res and res["std_error"]),
                    *[r["commitment"][x] for x in FIELDS]])

# Replays: the results files hold exactly the replays drawn from the banks.
drawn = {}
for s in pooled:
    cid = originals[s]["case_id"]
    for arm, folder in (("subsample-raw", "banks"), ("subsample-raw-ext1", "banks-ext1")):
        bank = json.loads((MAIN / folder / f"{cid}.{arm}.json").read_text(encoding="utf-8"))
        for d in bank["draws"]:
            key = hashlib.sha256(f"{cid}:{arm}:{d['stratum']}:{d['bank_index']}".encode()).hexdigest()[:32]
            drawn[key] = (s, COMPONENT[arm], d["stratum"], d["bank_index"])
recorded = {x["session_id"]: (x["slot"], x["component"], x["stratum"], x["bank_index"])
            for x in sessions if x["role"] == "replay" and x["sink"] == "results"}
assert recorded == drawn, (len(recorded), len(drawn))

with gzip.GzipFile(COHORT / "sessions.jsonl.gz", "wb", mtime=0) as gz:
    gz.write("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in sessions).encode())
with gzip.GzipFile(COHORT / "reports.jsonl.gz", "wb", mtime=0) as gz:
    gz.write("".join(json.dumps({"slot": s, "final_report": originals[s]["final_report"]}, ensure_ascii=False) + "\n"
                     for s in range(100)).encode())

# Reference banks: the ext1 bank keeps the base bank's scores and extends it.
bank = []
for s in sorted(pooled):
    cid = originals[s]["case_id"]
    b0 = json.loads((MAIN / "banks" / f"{cid}.subsample-raw.json").read_text(encoding="utf-8"))
    b1 = json.loads((MAIN / "banks-ext1" / f"{cid}.subsample-raw-ext1.json").read_text(encoding="utf-8"))
    assert b0["u"] == b1["u"] == originals[s]["u"] and all(b1["scores"][j] == p for j, p in b0["scores"].items())
    bank += [[s, int(j), g17(p)] for j, p in b1["scores"].items()]
bank.sort()
with gzip.GzipFile(INTER / "bank_pvalues.csv.gz", "wb", mtime=0) as gz, io.TextIOWrapper(gz, newline="") as t:
    w = csv.writer(t, lineterminator="\n")
    w.writerow(["slot", "bank_index", "p_value"])
    w.writerows(bank)

# Each original's frozen workspace (score.py and, for two slots, a notes file it wrote).
with tarfile.open(ARCHIVE) as t:
    for m in t.getmembers():
        if m.isfile():
            slot = int(m.name.split("/")[2])
            dest = COHORT / "programs" / f"slot-{slot:02d}" / Path(m.name).name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(t.extractfile(m).read())
print(len(sessions), "session records,", len(bank), "bank subsamples")
