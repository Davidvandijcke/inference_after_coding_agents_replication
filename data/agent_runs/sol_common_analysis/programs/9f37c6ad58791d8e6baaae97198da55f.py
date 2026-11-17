#!/usr/bin/env python3
"""Estimate the intention-to-treat effect on employment at 12 months.

Run: python score.py --data data.csv --output result.json
Requires numpy and scipy. The only model is OLS of employed_m12 on an
intercept, treatment, and indicators for the observed randomization sites.
Inference uses HC2 standard errors and a two-sided Student t-test with
n - rank(X) residual degrees of freedom, as specified in CODEBOOK.md.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import t


def analyze(data_path):
    with open(data_path, newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        required = {"site", "treatment", "employed_m12"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Data must contain site, treatment, and employed_m12.")
        rows = list(reader)
    if not rows:
        raise ValueError("Data contains no participants.")

    sites = np.array([row["site"].strip() for row in rows])
    treatment = np.array([float(row["treatment"]) for row in rows])
    outcome = np.array([float(row["employed_m12"]) for row in rows])
    if np.any(sites == ""):
        raise ValueError("Site must be observed for every participant.")
    if not np.isin(treatment, [0, 1]).all() or not np.isin(outcome, [0, 1]).all():
        raise ValueError("Treatment and employed_m12 must be observed binary values.")

    site_levels = sorted(set(sites))
    design = np.column_stack(
        [np.ones(len(rows)), treatment]
        + [(sites == site).astype(float) for site in site_levels[1:]]
    )
    n, k = design.shape
    if n <= k or np.linalg.matrix_rank(design) != k:
        raise ValueError("Model requires a full-rank design and positive residual df.")

    q, r = np.linalg.qr(design, mode="reduced")
    coefficients = np.linalg.solve(r, q.T @ outcome)
    residuals = outcome - design @ coefficients
    leverage = np.sum(q * q, axis=1)
    if np.any(1 - leverage <= np.finfo(float).eps):
        raise ValueError("HC2 is undefined for observations with unit leverage.")
    # Row 1 of (X'X)^(-1)X' gives each observation's treatment influence.
    influence = np.linalg.solve(r, q.T)[1]
    variance = np.sum(influence**2 * residuals**2 / (1 - leverage))
    std_error = float(np.sqrt(variance))
    if std_error <= 0 or not np.isfinite(std_error):
        raise ValueError("The treatment standard error must be positive and finite.")
    estimate = float(coefficients[1])
    p_value = float(2 * t.sf(abs(estimate / std_error), df=n - k))
    return {"estimate": estimate, "std_error": std_error, "p_value": p_value}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error("Output must be separate from the input data.")
    result = analyze(args.data)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
