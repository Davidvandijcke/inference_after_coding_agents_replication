"""Each usable original's u and tail and body replay counts; each program's rejection rate on new data.

A replay counts when classified and matches when its own verdict says so. Checks each replay's stratum against
the bank and the replayed subsamples against the run's seeded draw.

Inputs:  data/agent_runs/power_luna_login/sessions.jsonl.gz, data/intermediate/04_power/bank_p.csv.gz,
         program_draws.csv.gz
Outputs: data/derived/04_power/replay_counts.csv, program_benchmark.csv
"""
import csv
import gzip
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "data" / "agent_runs" / "power_luna_login"
INTER = ROOT / "data" / "intermediate" / "04_power"
OUT = ROOT / "data" / "derived" / "04_power"
TAIL, BODY = 40, 200  # replays planned per original and stratum

with gzip.open(RUNS / "sessions.jsonl.gz", "rt", encoding="utf-8") as f:
    sessions = [json.loads(line) for line in f]
originals = {r["slot"]: r for r in sessions if r["role"] == "original" and r["record"] == "result"}
usable = {slot: r for slot, r in originals.items() if r["p_value"] is not None}

bank = defaultdict(dict)
with gzip.open(INTER / "bank_p.csv.gz", "rt", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        bank[int(r["slot"])][int(r["bank_index"])] = float(r["p_value"]) if r["p_value"] else None

counts = {slot: {"x1": 0, "b1": 0, "x0": 0, "b0": 0} for slot in usable}
drawn = defaultdict(lambda: {"tail": set(), "body": set()})
for r in sessions:
    if r["role"] != "replay" or r["status"] != "classified":
        continue
    slot = r["slot"]
    u, p = usable[slot]["p_value"], bank[slot].get(r["bank_index"])
    where = f"slot {slot}, subsample {r['bank_index']}"
    assert p is not None, f"{where}: replayed, but the bank holds no p-value for it"
    stratum = "tail" if p <= u else "body"
    assert stratum == r["stratum"], f"{where}: the bank puts it in the {stratum}, the session record in the {r['stratum']}"
    assert r["bank_index"] not in drawn[slot][stratum], f"{where}: replayed more than once"
    drawn[slot][stratum].add(r["bank_index"])
    c = counts[slot]
    c["b1" if stratum == "tail" else "b0"] += 1
    c["x1" if stratum == "tail" else "x0"] += r["comparison"]["model_null_match"] is True

for slot, o in usable.items():
    tail = sorted(j for j, p in bank[slot].items() if p is not None and p <= o["p_value"])
    body = sorted(j for j, p in bank[slot].items() if p is not None and p > o["p_value"])
    pick = random.Random(f"{o['seed']}-subsample-raw-select")
    pick.shuffle(tail)
    pick.shuffle(body)
    assert drawn[slot] == {"tail": set(tail[:TAIL]), "body": set(body[:BODY])}, \
        f"slot {slot}: the replayed subsamples are not the first {TAIL} tail and {BODY} body of the seeded draw"

OUT.mkdir(parents=True, exist_ok=True)
with open(OUT / "replay_counts.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(["slot", "u", "x1", "b1", "x0", "b0", "b1_planned", "b0_planned"])
    for slot in sorted(counts):
        c = counts[slot]
        w.writerow([slot, repr(usable[slot]["p_value"]), c["x1"], c["b1"], c["x0"], c["b0"], TAIL, BODY])

draws = defaultdict(lambda: {"yes": [], "no": []})
with gzip.open(INTER / "program_draws.csv.gz", "rt", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        if r["p_value"]:
            draws[int(r["slot"])][r["effect"]].append(float(r["p_value"]))
with open(OUT / "program_benchmark.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(["slot", "effect_scored", "effect_reject_05", "null_scored", "null_reject_05"])
    for slot in sorted(draws):
        e, n = draws[slot]["yes"], draws[slot]["no"]
        w.writerow([slot, len(e), repr(sum(p <= 0.05 for p in e) / len(e)), len(n),
                    repr(sum(p <= 0.05 for p in n) / len(n))])
