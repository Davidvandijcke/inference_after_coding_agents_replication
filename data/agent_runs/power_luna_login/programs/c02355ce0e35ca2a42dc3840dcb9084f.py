#!/usr/bin/env python3
"""Site-adjusted ITT estimate for employment at 12 months."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import t


def score(data_path):
    with open(data_path, newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))

    if not rows:
        raise ValueError("The input dataset has no rows")
    required = {"site", "treatment", "employed_m12"}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    sites = sorted({row["site"] for row in rows})
    treatment = np.array([float(row["treatment"]) for row in rows])
    outcome = np.array([float(row["employed_m12"]) for row in rows])
    site_columns = [
        np.array([float(row["site"] == site) for row in rows])
        for site in sites[1:]
    ]
    design = np.column_stack([np.ones(len(rows)), treatment, *site_columns])

    n, k = design.shape
    if n <= k:
        raise ValueError("The dataset must have more observations than model parameters")

    xtx_inverse = np.linalg.pinv(design.T @ design)
    coefficients = xtx_inverse @ design.T @ outcome
    residuals = outcome - design @ coefficients
    leverage = np.sum((design @ xtx_inverse) * design, axis=1)
    weighted_design = design * (residuals**2 / (1.0 - leverage))[:, None]
    covariance = xtx_inverse @ design.T @ weighted_design @ xtx_inverse

    estimate = float(coefficients[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    degrees_freedom = n - k
    p_value = float(2.0 * t.sf(abs(estimate / std_error), degrees_freedom))
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
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
