#!/usr/bin/env python3
"""Estimate the full-sample intention-to-treat effect on employed_m12.

Usage: python score.py --data data.csv --output result.json
Requires numpy and scipy. The sole model is OLS of employed_m12 on an
intercept, treatment, and site indicators, with HC2 standard errors and a
two-sided Student t test using n - rank(X) residual degrees of freedom.
The estimate and standard error are in probability units, not percentages.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import t


def analyze(data_path):
    """Return only the three statistics for the fixed employment analysis."""
    with open(data_path, newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"treatment", "site", "employed_m12"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Data must include treatment, site, and employed_m12")
        rows = [(row["treatment"], row["site"], row["employed_m12"])
                for row in reader]
    if not rows:
        raise ValueError("Data must contain observations")

    treatment = np.array([float(row[0]) for row in rows])
    sites = np.array([row[1].strip() for row in rows])
    outcome = np.array([float(row[2]) for row in rows])
    if not np.isfinite(treatment).all() or not np.isfinite(outcome).all():
        raise ValueError("Treatment and outcome must be finite and fully observed")
    if set(treatment) != {0.0, 1.0}:
        raise ValueError("Treatment must contain both 0 and 1, and no other values")
    if not np.isin(outcome, [0.0, 1.0]).all() or (sites == "").any():
        raise ValueError("Employment must be binary and site must be observed")

    site_levels = sorted(set(sites))
    design = np.column_stack(
        [np.ones(len(rows)), treatment]
        + [(sites == level).astype(float) for level in site_levels[1:]]
    )
    n, k = design.shape
    if n <= k or np.linalg.matrix_rank(design) != k:
        raise ValueError("Model needs positive residual degrees of freedom and full rank")

    # QR avoids explicitly inverting X'X. HC2 divides squared residuals
    # by 1 - leverage before forming the sandwich covariance.
    q, r = np.linalg.qr(design, mode="reduced")
    coefficients = np.linalg.solve(r, q.T @ outcome)
    residuals = outcome - design @ coefficients
    leverage = np.sum(q * q, axis=1)
    if np.any(1.0 - leverage <= np.finfo(float).eps):
        raise ValueError("HC2 is undefined for an observation with leverage one")
    influence = np.linalg.solve(r, q.T)
    hc2_variance = np.sum(influence[1] ** 2 * residuals ** 2 / (1.0 - leverage))
    standard_error = float(np.sqrt(hc2_variance))
    if not np.isfinite(standard_error) or standard_error <= 0:
        raise ValueError("Treatment standard error must be finite and positive")
    estimate = float(coefficients[1])
    p_value = float(2.0 * t.sf(abs(estimate / standard_error), df=n - k))
    return {"estimate": estimate, "std_error": standard_error, "p_value": p_value}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error("Output must differ from input")
    result = analyze(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
