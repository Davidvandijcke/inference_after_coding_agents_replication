#!/usr/bin/env python3
"""Estimate the effect of assignment on 12-month employment.

Usage: python score.py --data data.csv --output result.json
Requires numpy and scipy. Fits OLS with an intercept, treatment, and site
indicators; reports the treatment coefficient, HC2 standard error, and
two-sided Student t p-value using n minus model rank degrees of freedom.
No outcome selection, subgroup selection, or additional analyses are run.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import t


def analyze(data_path):
    with open(data_path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"site", "treatment", "employed_m12"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Data must contain site, treatment, and employed_m12.")
        rows = list(reader)

    if not rows:
        raise ValueError("Data must contain participants.")
    sites = np.array([float(row["site"]) for row in rows])
    treatment = np.array([float(row["treatment"]) for row in rows])
    outcome = np.array([float(row["employed_m12"]) for row in rows])
    if not all(np.isfinite(values).all() for values in (sites, treatment, outcome)):
        raise ValueError("Analysis columns must be complete and finite.")
    if not np.isin(treatment, [0, 1]).all() or not np.isin(outcome, [0, 1]).all():
        raise ValueError("Treatment and employed_m12 must be binary (0 or 1).")

    levels = np.unique(sites)
    design = np.column_stack(
        [np.ones(len(rows)), treatment]
        + [(sites == level).astype(float) for level in levels[1:]]
    )
    n, k = design.shape
    if n <= k or np.linalg.matrix_rank(design) != k:
        raise ValueError("The site-adjusted treatment effect must be identifiable with positive residual degrees of freedom.")

    # Reduced QR avoids explicitly inverting X'X.
    q, r = np.linalg.qr(design, mode="reduced")
    coefficients = np.linalg.solve(r, q.T @ outcome)
    residuals = outcome - design @ coefficients
    leverage = np.sum(q * q, axis=1)
    if np.any(1 - leverage <= np.finfo(float).eps):
        raise ValueError("HC2 is undefined for an observation with unit leverage.")
    # Row 1 of (X'X)^(-1)X' gives each observation's weight in beta_treatment.
    treatment_weights = np.linalg.solve(r, q.T)[1]
    variance = np.sum(treatment_weights**2 * residuals**2 / (1 - leverage))
    std_error = float(np.sqrt(variance))
    if not np.isfinite(std_error) or std_error <= 0:
        raise ValueError("The treatment standard error must be finite and positive.")
    estimate = float(coefficients[1])
    p_value = float(2 * t.sf(abs(estimate / std_error), df=n - k))
    return {"estimate": estimate, "std_error": std_error, "p_value": p_value}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error("Output must differ from the input dataset.")
    result = analyze(args.data)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
