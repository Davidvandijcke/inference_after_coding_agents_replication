#!/usr/bin/env python3
"""Score the site-adjusted intention-to-treat effect on employment at 12 months."""

import argparse
import csv
import json
import sys

import numpy as np
from scipy.stats import t as student_t


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV dataset")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    with open(args.data, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("Input dataset has no observations")
    required = {"treatment", "site", "employed_m12"}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))

    sites = sorted({row["site"] for row in rows})
    # Site fixed effects; the first site is the reference category.
    x = np.asarray([
        [1.0, float(row["treatment"])]
        + [float(row["site"] == site) for site in sites[1:]]
        for row in rows
    ])
    y = np.asarray([float(row["employed_m12"]) for row in rows])
    n, k = x.shape
    if n <= k:
        raise ValueError("Not enough observations to estimate the model")

    xtx_inv = np.linalg.inv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    residual = y - x @ beta
    leverage = np.sum((x @ xtx_inv) * x, axis=1)
    meat = x.T @ (x * (residual**2 / (1.0 - leverage))[:, None])
    covariance = xtx_inv @ meat @ xtx_inv
    estimate = float(beta[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    statistic = estimate / std_error
    p_value = float(2.0 * student_t.sf(abs(statistic), df=n - k))

    result = {
        "estimate": estimate,
        "std_error": std_error,
        "p_value": p_value,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, np.linalg.LinAlgError) as exc:
        print(f"score.py: {exc}", file=sys.stderr)
        raise SystemExit(2)
