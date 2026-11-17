#!/usr/bin/env python3
"""Score the prespecified 12-month employment contrast in a Pathways dataset."""

import argparse
import csv
import json
import math

import numpy as np
from scipy.stats import t as student_t


def score(data_path):
    with open(data_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("The input dataset is empty")

    sites = sorted({row["site"] for row in rows})
    # Intercept, randomized assignment, and site indicators (first site omitted).
    design = []
    outcome = []
    for row in rows:
        treatment = float(row["treatment"])
        design.append([1.0, treatment] + [float(row["site"] == site) for site in sites[1:]])
        outcome.append(float(row["employed_m12"]))

    x = np.asarray(design, dtype=float)
    y = np.asarray(outcome, dtype=float)
    n, k = x.shape
    if n <= k:
        raise ValueError("The dataset has too few rows to estimate the model")

    xtx_inv = np.linalg.inv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    residual = y - x @ beta
    # HC2 uses each squared residual divided by one minus its OLS leverage.
    leverage = np.sum((x @ xtx_inv) * x, axis=1)
    adjusted_scores = x * (residual / np.sqrt(1.0 - leverage))[:, None]
    covariance = xtx_inv @ (adjusted_scores.T @ adjusted_scores) @ xtx_inv
    estimate = float(beta[1])
    std_error = float(math.sqrt(covariance[1, 1]))
    statistic = estimate / std_error
    p_value = float(2.0 * student_t.sf(abs(statistic), df=n - k))
    return {"estimate": estimate, "std_error": std_error, "p_value": p_value}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="CSV dataset with the codebook columns")
    parser.add_argument("--output", required=True, help="Path for the JSON result")
    args = parser.parse_args()
    result = score(args.data)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
