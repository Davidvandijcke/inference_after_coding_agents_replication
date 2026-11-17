#!/usr/bin/env python3
"""Frozen analysis: assignment effect on training_m12 among site 3 participants.

Usage: python score.py --data data.csv --output result.json
Requires scipy. Outputs probability units (0.10 means 10 percentage points).

Within one site the specified OLS model is outcome ~ intercept + treatment.
Its HC2 variance is s1**2/n1 + s0**2/n0, with sample variances s1**2
and s0**2. The two-sided t test uses OLS residual df n1+n0-2,
not Welch degrees of freedom. No outcomes, sites, or models are searched.
"""

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, variance

from scipy.stats import t


def chosen_analysis(data_path):
    groups = {0: [], 1: []}
    with open(data_path, newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        required = {"site", "treatment", "training_m12"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Missing required columns: site, treatment, training_m12")
        for row in reader:
            if float(row["site"]) != 3:
                continue
            assignment = float(row["treatment"])
            outcome = float(row["training_m12"])
            if assignment not in (0, 1) or outcome not in (0, 1):
                raise ValueError("Site 3 treatment and training_m12 must be binary")
            groups[int(assignment)].append(outcome)

    control, treated = groups[0], groups[1]
    if min(len(control), len(treated)) < 2:
        raise ValueError("Site 3 must contain at least two participants per arm")
    estimate = mean(treated) - mean(control)
    std_error = math.sqrt(variance(treated) / len(treated)
                          + variance(control) / len(control))
    df = len(treated) + len(control) - 2
    if std_error == 0:
        p_value = 1.0 if estimate == 0 else 0.0
    else:
        p_value = float(2 * t.sf(abs(estimate / std_error), df=df))
    return {"estimate": estimate, "std_error": std_error, "p_value": p_value}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error("Output must be separate from the input data")
    try:
        result = chosen_analysis(args.data)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8")


if __name__ == "__main__":
    main()
