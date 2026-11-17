#!/usr/bin/env python3
"""Score the 12-month employment treatment effect with site fixed effects."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t


def estimate(data_path):
    with open(data_path, newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    required = {"site", "treatment", "employed_m12"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("Input CSV must contain site, treatment, and employed_m12 columns")

    sites = np.array([row["site"] for row in rows])
    treatment = np.array([float(row["treatment"]) for row in rows])
    outcome = np.array([float(row["employed_m12"]) for row in rows])
    site_levels = sorted(set(sites))

    # Intercept, treatment assignment, and site indicators (first site omitted).
    design = np.column_stack(
        [np.ones(len(rows)), treatment]
        + [(sites == level).astype(float) for level in site_levels[1:]]
    )
    n, k = design.shape
    if n <= k:
        raise ValueError("Not enough observations to estimate the model")

    xtx_inv = np.linalg.inv(design.T @ design)
    coefficients = xtx_inv @ design.T @ outcome
    residuals = outcome - design @ coefficients

    # HC2 sandwich covariance: each squared residual is divided by (1 - h_ii).
    leverage = np.einsum("ij,jk,ik->i", design, xtx_inv, design)
    if np.any(1.0 - leverage <= 0):
        raise ValueError("HC2 variance is undefined for this design")
    weighted_design = design * (residuals**2 / (1.0 - leverage))[:, None]
    covariance = xtx_inv @ design.T @ weighted_design @ xtx_inv

    estimate = float(coefficients[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    statistic = estimate / std_error
    p_value = float(2.0 * student_t.sf(abs(statistic), df=n - k))
    return {"estimate": estimate, "std_error": std_error, "p_value": p_value}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to input CSV")
    parser.add_argument("--output", required=True, help="Path for JSON results")
    args = parser.parse_args()

    result = estimate(args.data)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
