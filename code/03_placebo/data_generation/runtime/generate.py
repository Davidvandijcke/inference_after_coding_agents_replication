"""Placebo rebuild of a real randomized experiment.

Replaces the simulation's generate.py. The base file of an experiment holds its real
baseline variables and outcomes and no assignment of any kind. Each seed draws a fresh
treatment within blocks at the experiment's frozen treated count, a fresh set of
participant ids and a fresh row order. The draw uses nothing but the seed and the block,
so treatment is independent of every outcome and every null is exactly true.

The experiment is named by n, because the base files have different sizes; this keeps
the runner's fresh spec ({seed, n}) and make_references.py's calls unchanged.

Usage: python generate.py --seed SEED --n N --out data.csv
"""
import argparse
import csv
import json
import threading
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXPERIMENTS = json.loads((HERE / "base" / "experiments.json").read_text())
BY_N = {e["n"]: name for name, e in EXPERIMENTS.items()}
_LOCK = threading.Lock()
_BASE = {}


def load_base(name):
    with _LOCK:
        if name not in _BASE:
            with open(HERE / "base" / f"{name}_base.csv", newline="") as f:
                rows = list(csv.reader(f))
            header = rows[0]
            body = np.empty((len(rows) - 1, len(header)), dtype=object)
            for i, r in enumerate(rows[1:]):
                body[i, :] = r
            _BASE[name] = (header, body, body[:, header.index("block")].astype(int))
        return _BASE[name]


def generate(seed, n):
    name = BY_N[n]
    header, body, block = load_base(name)
    rng = np.random.default_rng(seed)
    T = np.zeros(n, dtype=int)
    for b, k in sorted(EXPERIMENTS[name]["treated_per_block"].items(), key=lambda kv: int(kv[0])):
        idx = np.flatnonzero(block == int(b))
        T[rng.choice(idx, size=k, replace=False)] = 1
    ids = rng.permutation(np.arange(100001, 100001 + n))
    order = rng.permutation(n)
    cols = {"participant_id": ids[order], "block": block[order], "treatment": T[order]}
    for j, v in enumerate(header):
        if v != "block":
            cols[v] = body[order, j]
    return cols


def write_csv(cols, path, rows=None):
    names = list(cols)
    idx = range(len(cols[names[0]])) if rows is None else rows
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(names)
        for i in idx:
            w.writerow([cols[k][i] for k in names])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    write_csv(generate(a.seed, a.n), a.out)
