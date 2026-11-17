#!/usr/bin/env python3
"""Regenerate the realistic request's reference bank from the restricted ISSP file. No model calls.

Runs the original agent's program (data/agent_runs/issp_request_luna/original/programs/analysis_final.py) on the full
analysis file and on each of the 4,000 bank subsamples of 16 of the 46 countries (all respondents of each chosen
country, rows in a seeded random order), exactly as the experiment's runner wrote each subsample and each replay's
data.csv. Writes, in the folder given by --out, original_full_data.json (the program's output on the full file) and
bank_p.csv (one row per subsample: its countries, the SHA-256 of the subsample file and the program's p-value), in
the format of the shipped files in data/intermediate/06_immigration_request/, which they can be compared with.

Usage, from the package root, with the file built by code/05_immigration_personas/data_generation/build_issp_file.py:
    python code/06_immigration_request/data_generation/score_bank.py --data <issp_clean.csv> \
        --out data/derived/06_immigration_request/bank_check [--first N]
The program needs pandas, statsmodels and scipy. It takes about 1.6 seconds per subsample on one core, so about
2 hours for the full bank (21 minutes with 30 parallel workers on the cluster); --first N scores only the first N
subsamples, as a test.
"""
import argparse
import csv
import hashlib
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PROGRAM = ROOT / "data/agent_runs/issp_request_luna/original/programs/analysis_final.py"
SEED, K, BANK = "issp-luna-20261027", 16, 4000


def run_program(path, tmp):
    out = Path(tmp) / "out.json"
    subprocess.run([sys.executable, str(PROGRAM), "--data", str(path), "--output", str(out)], check=True,
                   capture_output=True)
    return json.loads(out.read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--first", type=int, default=BANK)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    header, *lines = a.data.read_text(encoding="utf-8").splitlines()
    col = header.split(",").index("iso_country")
    by = {}
    for line in lines:
        by.setdefault(line.split(",")[col], []).append(line)
    countries = sorted(by, key=int)
    a.out.mkdir(parents=True, exist_ok=True)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        full = run_program(a.data, tmp)
        (a.out / "original_full_data.json").write_text(json.dumps(full, indent=1, ensure_ascii=False) + "\n",
                                                       encoding="utf-8")
        sub = Path(tmp) / "data.csv"
        for j in range(a.first):
            chosen = random.Random(f"{SEED}-bank-{j}").sample(countries, K)
            body = [line for c in chosen for line in by[c]]
            random.Random(f"{SEED}-order-{j}").shuffle(body)
            text = "\n".join([header] + body) + "\n"
            sub.write_text(text, encoding="utf-8")
            rows.append({"bank_index": j, "countries": " ".join(sorted(chosen, key=int)),
                         "data_sha256": hashlib.sha256(text.encode()).hexdigest(),
                         "p": repr(float(run_program(sub, tmp)["p_value"]))})
    with open(a.out / "bank_p.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
