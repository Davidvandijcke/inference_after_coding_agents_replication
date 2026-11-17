#!/usr/bin/env python3
"""Score the site-adjusted 12-month employment treatment effect."""

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
    # Intercept, treatment, and site indicators (first site is the reference).
    x = np.asarray(
        [[1.0, float(row["treatment"])] + [float(row["site"] == site) for site in sites[1:]]
         for row in rows],
        dtype=float,
    )
    y = np.asarray([float(row["employed_m12"]) for row in rows], dtype=float)

    xtx_inv = np.linalg.inv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    residuals = y - x @ beta
    # HC2 sandwich covariance: each squared residual is divided by (1 - h_ii).
    hat = np.einsum("ij,jk,ik->i", x, xtx_inv, x)
    scaled_x = x * (residuals / np.sqrt(1.0 - hat))[:, None]
    covariance = xtx_inv @ (scaled_x.T @ scaled_x) @ xtx_inv

    estimate = float(beta[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    degrees_freedom = len(y) - x.shape[1]
    statistic = estimate / std_error
    p_value = float(2.0 * t.sf(abs(statistic), degrees_freedom))
    return {"estimate": estimate, "std_error": std_error, "p_value": p_value}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="CSV dataset")
    parser.add_argument("--output", required=True, help="Path for JSON result")
    args = parser.parse_args()
    result = score(args.data)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as destination:
        json.dump(result, destination, indent=2)
        destination.write("\n")


if __name__ == "__main__":
    main()
