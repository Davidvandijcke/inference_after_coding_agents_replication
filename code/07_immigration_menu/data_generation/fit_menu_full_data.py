#!/usr/bin/env python3
"""Fit the 108 menu specifications on the full restricted ISSP file with the agents' own helper. No model calls.

Usage, from the package root (seconds; needs pandas, numpy and statsmodels):
    python code/07_immigration_menu/data_generation/fit_menu_full_data.py <issp_clean.csv> <output folder>
Runs data/agent_runs/issp_menu_anti/inputs/catalogue_helper.py --all on the file and writes
<output folder>/menu_full_data.csv in the format of the shipped data/intermediate/07_immigration_menu/menu_full_data.csv:
per specification its outcome, exposure, coverage rule and controls, the number of countries, the country-level OLS
slope, its HC3 standard error, t-statistic and p-value.
"""
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
INPUTS = ROOT / "data/agent_runs/issp_menu_anti/inputs"
FIELDS = ["model_id", "outcome", "exposure", "coverage", "control_set", "n_countries", "slope", "hc3_se", "hc3_t",
          "raw_hc3_p"]

with tempfile.TemporaryDirectory() as tmp:
    out = Path(tmp) / "fits.json"
    subprocess.run([sys.executable, str(INPUTS / "catalogue_helper.py"), "--catalogue", str(INPUTS / "CATALOGUE.json"),
                    "--data", sys.argv[1], "--all", "--output", str(out)], check=True, capture_output=True)
    fits = json.loads(out.read_text(encoding="utf-8"))["fits"]
path = Path(sys.argv[2]) / "menu_full_data.csv"
path.parent.mkdir(parents=True, exist_ok=True)
with open(path, "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh, lineterminator="\n")
    w.writerow(FIELDS)
    w.writerows([f[k] if not isinstance(f[k], float) else repr(f[k]) for k in FIELDS]
                for f in sorted(fits, key=lambda f: f["sequence"]))
