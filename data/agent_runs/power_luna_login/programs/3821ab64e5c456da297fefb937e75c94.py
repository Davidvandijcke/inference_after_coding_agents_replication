#!/usr/bin/env python3
"""Score the site-adjusted intention-to-treat effect on employment at 12 months."""

import argparse
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


def estimate(data_path):
    data = pd.read_csv(data_path)
    required = {"site", "treatment", "employed_m12"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    if data[list(required)].isna().any().any():
        raise ValueError("Required analysis columns contain missing values")

    site_codes, site_levels = pd.factorize(data["site"], sort=True)
    if len(site_levels) < 1:
        raise ValueError("No site observations found")
    # Intercept, treatment, and indicators for all sites except the first.
    site_indicators = np.eye(len(site_levels), dtype=float)[site_codes, 1:]
    x = np.column_stack(
        [np.ones(len(data)), data["treatment"].to_numpy(dtype=float), site_indicators]
    )
    y = data["employed_m12"].to_numpy(dtype=float)
    n, p = x.shape
    if n <= p:
        raise ValueError("Insufficient observations for the regression")

    xtx_inv = np.linalg.inv(x.T @ x)
    beta = xtx_inv @ (x.T @ y)
    residuals = y - x @ beta
    # HC2: each squared residual is adjusted by (1 - leverage).
    leverage = np.einsum("ij,jk,ik->i", x, xtx_inv, x)
    if np.any(1.0 - leverage <= 0):
        raise ValueError("HC2 variance is undefined for this design")
    weighted_x = x * (residuals**2 / (1.0 - leverage))[:, None]
    covariance = xtx_inv @ (x.T @ weighted_x) @ xtx_inv

    effect = float(beta[1])
    standard_error = float(np.sqrt(covariance[1, 1]))
    statistic = effect / standard_error
    p_value = float(2.0 * student_t.sf(abs(statistic), df=n - p))
    return {
        "estimate": effect,
        "std_error": standard_error,
        "p_value": p_value,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Input CSV with evaluation columns")
    parser.add_argument("--output", required=True, help="Path for JSON result")
    args = parser.parse_args()
    result = estimate(args.data)
    with open(args.output, "w", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"score.py: {exc}", file=sys.stderr)
        raise SystemExit(1)
