#!/usr/bin/env python3
"""Estimate the full-sample effect on employed_m12, adjusting for site.

Usage: python score.py --data data.csv --output results.json
Dependencies: numpy, statsmodels.
The estimate and standard error are in probability units, not percentage points.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import statsmodels.api as sm


def analyze(data_path):
    """Run only OLS of employed_m12 on assignment and site indicators."""
    with Path(data_path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"site", "treatment", "employed_m12"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Data must include site, treatment, and employed_m12.")
        rows = [
            (row["site"].strip(), float(row["treatment"]), float(row["employed_m12"]))
            for row in reader
        ]
    if not rows or any(not row[0] for row in rows):
        raise ValueError("Data must contain participants with observed sites.")

    site = np.array([row[0] for row in rows])
    treatment = np.array([row[1] for row in rows])
    outcome = np.array([row[2] for row in rows])
    if not np.isin(treatment, [0, 1]).all() or not np.isin(outcome, [0, 1]).all():
        raise ValueError("Treatment and employed_m12 must be observed binary values.")

    levels = sorted(set(site))
    design = np.column_stack(
        [np.ones(len(rows)), treatment]
        + [(site == level).astype(float) for level in levels[1:]]
    )
    if len(rows) <= design.shape[1] or np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("Insufficient variation or observations for the specified regression.")
    fit = sm.OLS(outcome, design).fit(cov_type="HC2", use_t=True)
    result = {
        "estimate": float(fit.params[1]),
        "std_error": float(fit.bse[1]),
        "p_value": float(fit.pvalues[1]),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("The specified HC2 t-test is undefined for these data.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error("The output path must differ from the input data path.")
    result = analyze(args.data)
    args.output.write_text(json.dumps(result, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
