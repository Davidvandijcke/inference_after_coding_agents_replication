#!/usr/bin/env python3
"""Estimate the 12-month employment effect among women, adjusted for site."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import t


def estimate(data_path):
    with open(data_path, newline="", encoding="utf-8") as source:
        rows = [row for row in csv.DictReader(source) if int(row["female"]) == 1]

    if not rows:
        raise ValueError("No female participants found in the input data")

    sites = sorted({int(row["site"]) for row in rows})
    site_values = np.array([int(row["site"]) for row in rows])
    treatment = np.array([float(row["treatment"]) for row in rows])
    outcome = np.array([float(row["employed_m12"]) for row in rows])

    # Intercept, treatment, and site indicators (first site is the reference).
    columns = [np.ones(len(rows)), treatment]
    columns.extend((site_values == site).astype(float) for site in sites[1:])
    design = np.column_stack(columns)

    xtx_inverse = np.linalg.inv(design.T @ design)
    coefficients = xtx_inverse @ design.T @ outcome
    residuals = outcome - design @ coefficients
    leverage = np.sum((design @ xtx_inverse) * design, axis=1)
    weighted_design = design * (residuals**2 / (1.0 - leverage))[:, None]
    covariance = xtx_inverse @ design.T @ weighted_design @ xtx_inverse

    estimate = float(coefficients[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    degrees_freedom = len(rows) - design.shape[1]
    p_value = float(2.0 * t.sf(abs(estimate / std_error), degrees_freedom))
    return {
        "estimate": estimate,
        "std_error": std_error,
        "p_value": p_value,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Score the site-adjusted 12-month employment effect among women."
    )
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    result = estimate(args.data)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
