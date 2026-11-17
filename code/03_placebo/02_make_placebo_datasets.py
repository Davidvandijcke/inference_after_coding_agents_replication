"""The 102 placebo datasets, checked with every replay subsample against the data the agents received.

Slot s uses study ("wellness", "teachers", "consumers")[s % 3] and seed 9,000,000 + s: a treatment drawn within each
block at the block's treated count in experiments.json, new participant ids and a new row order (the cohort's
generate.py). Base files, datasets and subsamples are checked against the SHA-256 recorded when the cohort ran.

Inputs:  data/derived/03_placebo/base/, data/agent_runs/placebo_luna/recorded_sha256.csv, sessions.jsonl.gz
Outputs: data/derived/03_placebo/datasets/original_NNN.csv
"""
import csv
import gzip
import hashlib
import io
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "data" / "agent_runs" / "placebo_luna"
DERIVED = ROOT / "data" / "derived" / "03_placebo"
BASE = DERIVED / "base"
STUDIES = ("wellness", "teachers", "consumers")
EXPERIMENTS = json.loads((BASE / "experiments.json").read_text(encoding="utf-8"))


def load_base(study):
    with open(BASE / f"{study}_base.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    body = np.empty((len(rows) - 1, len(rows[0])), dtype=object)
    for i, r in enumerate(rows[1:]):
        body[i, :] = r
    return rows[0], body, body[:, rows[0].index("block")].astype(int)


def generate(seed, header, body, block, treated_per_block):
    n = len(body)
    rng = np.random.default_rng(seed)
    treat = np.zeros(n, dtype=int)
    for b, k in sorted(treated_per_block.items(), key=lambda kv: int(kv[0])):
        treat[rng.choice(np.flatnonzero(block == int(b)), size=k, replace=False)] = 1
    ids = rng.permutation(np.arange(100001, 100001 + n))
    order = rng.permutation(n)
    cols = {"participant_id": ids[order], "block": block[order], "treatment": treat[order]}
    for j, v in enumerate(header):
        if v != "block":
            cols[v] = body[order, j]
    return [list(cols)] + [[cols[k][i] for k in cols] for i in range(n)]


def csv_bytes(rows):
    buf = io.StringIO(newline="")
    csv.writer(buf).writerows(rows)
    return buf.getvalue().encode()


def sha(content):
    return hashlib.sha256(content).hexdigest()


def require(ok, message):  # not assert, which python -O skips
    if not ok:
        sys.exit(message)


with open(RUNS / "recorded_sha256.csv", newline="", encoding="utf-8") as f:
    recorded = {r["file"]: r["sha256"] for r in csv.DictReader(f)}
for name in ("wellness_base.csv", "teachers_base.csv", "consumers_base.csv", "experiments.json"):
    require(sha((BASE / name).read_bytes()) == recorded[f"base/{name}"],
            f"base/{name} differs from the file the agents' runtime held")

sessions = [json.loads(l) for l in gzip.open(RUNS / "sessions.jsonl.gz", "rt", encoding="utf-8")]
served = {r["slot"]: r["data_sha256"] for r in sessions if r["role"] == "original"}
(DERIVED / "datasets").mkdir(parents=True, exist_ok=True)
bases = {s: load_base(s) for s in STUDIES}
data = {}
for slot in range(102):
    study = STUDIES[slot % 3]
    data[slot] = generate(9_000_000 + slot, *bases[study], EXPERIMENTS[study]["treated_per_block"])
    content = csv_bytes(data[slot])
    require(sha(content) == served[slot], f"slot {slot}: dataset differs from the data the original agent received")
    (DERIVED / "datasets" / f"original_{slot:03d}.csv").write_bytes(content)

replays = 0
for r in sessions:
    if r["role"] != "replay":
        continue
    rows, s = data[r["slot"]], r["slot"]
    n = len(rows) - 1
    idx = random.Random(f"{9_000_000 + s}-bank-{r['bank_index']}").sample(range(n), int(0.10 * n + 0.5))
    require(sha(csv_bytes([rows[0]] + [rows[1 + i] for i in idx])) == r["data_sha256"],
            f"replay {r['session_id']} (slot {s}, subsample {r['bank_index']}): data differ from the data the replay received")
    replays += 1
print(f"3 base files, 102 datasets and {replays} replay subsamples match the data the agents received")
