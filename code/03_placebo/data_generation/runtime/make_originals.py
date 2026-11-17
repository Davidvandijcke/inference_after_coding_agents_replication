"""Build the original datasets and their case list on the cluster (one CSV per original).

Slots cycle through the experiments so that any stage made of a slot range is balanced.

Usage: python make_originals.py EXP_ROOT [--slots 51] [--first-slot 0] [--seed-base 9000000] [--prefix main] [--list originals.jsonl]
"""
import argparse
import json
import secrets
from pathlib import Path

import generate

ORDER = ("wellness", "teachers", "consumers")

ap = argparse.ArgumentParser()
ap.add_argument("exp")
ap.add_argument("--slots", type=int, default=51)
ap.add_argument("--first-slot", type=int, default=0)
ap.add_argument("--seed-base", type=int, default=9_000_000)
ap.add_argument("--prefix", default="main")
ap.add_argument("--list", default="originals.jsonl")
a = ap.parse_args()

exp = Path(a.exp)
(exp / "data" / "originals").mkdir(parents=True, exist_ok=True)
cases = []
for slot in range(a.first_slot, a.first_slot + a.slots):
    experiment = ORDER[slot % len(ORDER)]
    n = generate.EXPERIMENTS[experiment]["n"]
    seed = a.seed_base + slot
    name = f"{a.prefix}-{slot:02d}.csv"
    generate.write_csv(generate.generate(seed, n), exp / "data" / "originals" / name)
    cases.append({"case_id": secrets.token_hex(16), "kind": "original",
                  "data_path": str(exp / "data" / "originals" / name),
                  "meta": {"slot": slot, "seed": seed, "experiment": experiment, "n": n}})
with open(exp / a.list, "w") as f:
    for c in cases:
        f.write(json.dumps(c) + "\n")
print(len(cases), "originals ->", a.list)
