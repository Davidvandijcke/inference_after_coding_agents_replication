#!/usr/bin/env python3
"""Estimate the assignment effect on food insecurity at 12 months.

Usage: python score.py --data data.csv --output result.json
Dependencies: numpy and statsmodels (including its scipy dependency).

Fits only food_insecure_m12 ~ treatment + categorical site indicators,
with an intercept, HC2 standard errors, and a two-sided t-test using
the regression residual degrees of freedom. The p-value is nominal;
it does not account for this endpoint's exploratory selection.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import statsmodels.api as sm


def score(data_path):
    with open(data_path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"site", "treatment", "food_insecure_m12"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Data must contain: " + ", ".join(sorted(required)))
        rows = [(row["site"], float(row["treatment"]),
                 float(row["food_insecure_m12"])) for row in reader]

    if not rows:
        raise ValueError("Data must contain participants.")
    sites = np.array([row[0] for row in rows])
    treatment = np.array([row[1] for row in rows])
    outcome = np.array([row[2] for row in rows])
    if np.any(sites == ""):
        raise ValueError("Site must be observed for every participant.")
    if not np.isin(treatment, [0, 1]).all() or not np.isin(outcome, [0, 1]).all():
        raise ValueError("Treatment and food_insecure_m12 must be observed binary values.")

    levels = sorted(set(sites))
    design = np.column_stack(
        [np.ones(len(rows)), treatment]
        + [(sites == level).astype(float) for level in levels[1:]]
    )
    if len(rows) <= design.shape[1] or np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("Insufficient variation to estimate treatment and site effects.")

    fit = sm.OLS(outcome, design).fit(cov_type="HC2", use_t=True)
    result = {
        "estimate": float(fit.params[1]),
        "std_error": float(fit.bse[1]),
        "p_value": float(fit.pvalues[1]),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("The data do not support finite HC2 inference.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error("--output must differ from --data")
    result = score(args.data)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, allow_nan=False, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
