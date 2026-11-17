#!/usr/bin/env python3
"""Score the site-adjusted treatment effect on employment at 12 months."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import t


def estimate(data_path):
    with open(data_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    required = {"site", "treatment", "employed_m12"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("Input must contain site, treatment, and employed_m12 columns")

    sites = sorted({row["site"] for row in rows})
    y = np.asarray([float(row["employed_m12"]) for row in rows])
    treatment = np.asarray([float(row["treatment"]) for row in rows])
    site_values = np.asarray([row["site"] for row in rows])
    site_dummies = np.column_stack(
        [(site_values == site).astype(float) for site in sites[1:]]
    ) if len(sites) > 1 else np.empty((len(rows), 0))
    X = np.column_stack((np.ones(len(rows)), treatment, site_dummies))

    n, k = X.shape
    if n <= k or np.linalg.matrix_rank(X) < k:
        raise ValueError("Design matrix must have full column rank and residual degrees of freedom")

    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    residuals = y - X @ beta
    leverage = np.sum((X @ xtx_inv) * X, axis=1)
    hc2_meat = X.T @ ((residuals**2 / (1.0 - leverage))[:, None] * X)
    covariance = xtx_inv @ hc2_meat @ xtx_inv

    estimate = float(beta[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    df = n - k
    p_value = float(2.0 * t.sf(abs(estimate / std_error), df))
    return {"estimate": estimate, "std_error": std_error, "p_value": p_value}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="CSV dataset path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()
    result = estimate(args.data)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
