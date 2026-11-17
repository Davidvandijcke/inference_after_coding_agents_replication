#!/usr/bin/env python3
"""Score the Pathways to Work 12-month employment result.

Usage: python score.py --data <path> --output <path>

Fits employment at 12 months on randomized assignment and site indicators,
then reports the treatment coefficient, HC2 standard error, and two-sided
t-test p-value. The analysis uses no baseline covariates.
"""

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="CSV dataset")
    parser.add_argument("--output", required=True, help="JSON output path")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = {"site", "treatment", "employed_m12"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    if data[list(required)].isna().any().any():
        raise ValueError("Required analysis columns must not contain missing values")

    treatment = data["treatment"].to_numpy(dtype=float)
    sites = pd.get_dummies(data["site"], drop_first=True, dtype=float).to_numpy()
    design = np.column_stack((np.ones(len(data)), treatment, sites))
    outcome = data["employed_m12"].to_numpy(dtype=float)
    n, k = design.shape
    if n <= k or np.linalg.matrix_rank(design) < k:
        raise ValueError("Design matrix must have full rank and positive residual degrees of freedom")

    bread = np.linalg.inv(design.T @ design)
    coefficients = bread @ design.T @ outcome
    residuals = outcome - design @ coefficients
    leverage = np.sum((design @ bread) * design, axis=1)
    hc2_meat = design.T @ ((residuals**2 / (1.0 - leverage))[:, None] * design)
    covariance = bread @ hc2_meat @ bread

    estimate = float(coefficients[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    statistic = estimate / std_error
    p_value = float(2.0 * student_t.sf(abs(statistic), df=n - k))
    result = {
        "estimate": estimate,
        "std_error": std_error,
        "p_value": p_value,
    }
    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2)
        output_file.write("\n")


if __name__ == "__main__":
    main()
