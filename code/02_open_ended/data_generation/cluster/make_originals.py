"""Generate original datasets and a case list (run locally, then sync).

Usage: python make_originals.py OUT_DIR EXP_ROOT [--n 32] [--seed-base 5000000] [--prefix orig] [--list originals.jsonl]
"""
import argparse
import json
import secrets
from pathlib import Path

import generate

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("exp")
ap.add_argument("--n", type=int, default=32)
ap.add_argument("--seed-base", type=int, default=5_000_000)  # pilot namespace
ap.add_argument("--prefix", default="orig")
ap.add_argument("--list", default="originals.jsonl")
a = ap.parse_args()

out = Path(a.out)
(out / "data" / "originals").mkdir(parents=True, exist_ok=True)
cases = []
for slot in range(a.n):
    seed = a.seed_base + slot
    name = f"{a.prefix}-{slot:02d}.csv"
    generate.write_csv(generate.generate(seed, 2000), out / "data" / "originals" / name)
    cases.append({"case_id": secrets.token_hex(16), "kind": "original",
                  "data_path": f"{a.exp}/data/originals/{name}", "meta": {"slot": slot, "seed": seed}})
with open(out / a.list, "w") as f:
    for c in cases:
        f.write(json.dumps(c) + "\n")
print(len(cases), "originals ->", a.list)
