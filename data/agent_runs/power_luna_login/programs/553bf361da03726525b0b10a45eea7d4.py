#!/usr/bin/env python3
"""Score the site-adjusted 12-month employment effect in a Pathways dataset."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.special import stdtr


def estimate_effect(data_path):
    with Path(data_path).open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))

    required = {"site", "treatment", "employed_m12"}
    if not rows:
        raise ValueError("The input dataset has no rows")
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    sites = np.array([int(row["site"]) for row in rows])
    treatment = np.array([float(row["treatment"]) for row in rows])
    outcome = np.array([float(row["employed_m12"]) for row in rows])
    site_values = sorted(set(sites))
    site_indicators = [
        (sites == site).astype(float) for site in site_values[1:]
    ]
    design = np.column_stack([np.ones(len(rows)), treatment, *site_indicators])

    n, k = design.shape
    if n <= k:
        raise ValueError("Not enough observations to estimate the model")

    xtx_inverse = np.linalg.inv(design.T @ design)
    coefficients = xtx_inverse @ design.T @ outcome
    residuals = outcome - design @ coefficients
    leverage = np.sum((design @ xtx_inverse) * design, axis=1)
    weighted_x = design * (residuals**2 / (1.0 - leverage))[:, None]
    covariance = xtx_inverse @ design.T @ weighted_x @ xtx_inverse

    estimate = float(coefficients[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    t_statistic = estimate / std_error
    p_value = float(2.0 * stdtr(n - k, -abs(t_statistic)))
    return {
        "estimate": estimate,
        "std_error": std_error,
        "p_value": p_value,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV dataset")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    result = estimate_effect(args.data)
    with Path(args.output).open("w", encoding="utf-8") as destination:
        json.dump(result, destination, indent=2)
        destination.write("\n")


if __name__ == "__main__":
    main()
