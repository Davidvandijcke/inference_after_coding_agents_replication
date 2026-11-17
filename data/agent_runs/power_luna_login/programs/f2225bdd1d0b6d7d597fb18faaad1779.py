#!/usr/bin/env python3
"""Score the site-adjusted intent-to-treat effect on 12-month employment."""

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import t


def score(data_path):
    data = pd.read_csv(data_path)
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
    if n <= k:
        raise ValueError("Insufficient observations to estimate the model")

    xtx_inv = np.linalg.inv(design.T @ design)
    coefficients = xtx_inv @ design.T @ outcome
    residuals = outcome - design @ coefficients
    leverage = np.sum((design @ xtx_inv) * design, axis=1)
    adjusted_residual_variance = residuals**2 / (1.0 - leverage)
    meat = design.T @ (adjusted_residual_variance[:, None] * design)
    covariance = xtx_inv @ meat @ xtx_inv

    estimate = float(coefficients[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    statistic = estimate / std_error
    p_value = float(2.0 * t.sf(abs(statistic), df=n - k))
    return {
        "estimate": estimate,
        "std_error": std_error,
        "p_value": p_value,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV dataset")
    parser.add_argument("--output", required=True, help="Path for the JSON result")
    args = parser.parse_args()
    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(score(args.data), output_file, indent=2)
        output_file.write("\n")


if __name__ == "__main__":
    main()
