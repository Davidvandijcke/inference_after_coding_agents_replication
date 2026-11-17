#!/usr/bin/env python3
"""Score the site-adjusted treatment effect on 12-month employment."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t


def estimate(data_path):
    with open(data_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError("The input dataset has no rows")
    required = {"site", "treatment", "employed_m12"}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))

    sites = sorted({row["site"] for row in rows})
    # Intercept, treatment, and site indicators (first site is the reference).
    x = np.empty((len(rows), 2 + max(0, len(sites) - 1)), dtype=float)
    y = np.empty(len(rows), dtype=float)
    site_index = {site: i for i, site in enumerate(sites)}
    for i, row in enumerate(rows):
        x[i, 0] = 1.0
        x[i, 1] = float(row["treatment"])
        for j, site in enumerate(sites[1:], start=2):
            x[i, j] = float(row["site"] == site)
        y[i] = float(row["employed_m12"])

    n, k = x.shape
    if n <= k:
        raise ValueError("Not enough rows to estimate the regression")

    xtx_inv = np.linalg.inv(x.T @ x)
    beta = xtx_inv @ (x.T @ y)
    residual = y - x @ beta
    # HC2 sandwich covariance: each squared residual is divided by (1 - h_ii).
    leverage = np.sum((x @ xtx_inv) * x, axis=1)
    meat = x.T @ ((residual ** 2 / (1.0 - leverage))[:, None] * x)
    covariance = xtx_inv @ meat @ xtx_inv
    se = float(np.sqrt(covariance[1, 1]))
    estimate = float(beta[1])
    statistic = estimate / se
    p_value = float(2.0 * student_t.sf(abs(statistic), df=n - k))
    return {"estimate": estimate, "std_error": se, "p_value": p_value}


def main():
    parser = argparse.ArgumentParser(
        description="Estimate the site-adjusted treatment effect on 12-month employment."
    )
    parser.add_argument("--data", required=True, help="Input CSV dataset")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    result = estimate(args.data)
    output_path = Path(args.output)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
