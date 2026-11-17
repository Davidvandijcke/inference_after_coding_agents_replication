#!/usr/bin/env python3
"""Score the site-adjusted intention-to-treat effect on 12-month employment."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t


def score(data_path):
    with open(data_path, newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Input data has no participant rows")

    required = {"site", "treatment", "employed_m12"}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    sites = np.array([int(row["site"]) for row in rows])
    treatment = np.array([float(row["treatment"]) for row in rows])
    outcome = np.array([float(row["employed_m12"]) for row in rows])
    levels = sorted(set(sites))
    site_indicators = np.column_stack(
        [(sites == level).astype(float) for level in levels[1:]]
    ) if len(levels) > 1 else np.empty((len(rows), 0))

    # OLS with an intercept, treatment assignment, and site fixed effects.
    design = np.column_stack((np.ones(len(rows)), treatment, site_indicators))
    n, k = design.shape
    if n <= k:
        raise ValueError("Insufficient rows to estimate the model")

    xtx_inverse = np.linalg.inv(design.T @ design)
    coefficients = xtx_inverse @ design.T @ outcome
    residuals = outcome - design @ coefficients
    leverage = np.sum((design @ xtx_inverse) * design, axis=1)

    # HC2 sandwich covariance: each squared residual is divided by 1 - leverage.
    weighted_design = design * (residuals**2 / (1.0 - leverage))[:, None]
    covariance = xtx_inverse @ design.T @ weighted_design @ design @ xtx_inverse
    estimate = float(coefficients[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    statistic = estimate / std_error
    p_value = float(2.0 * student_t.sf(abs(statistic), df=n - k))

    return {
        "estimate": estimate,
        "std_error": std_error,
        "p_value": p_value,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Estimate the site-adjusted treatment effect on employed_m12."
    )
    parser.add_argument("--data", required=True, help="Input CSV dataset")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    result = score(args.data)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
