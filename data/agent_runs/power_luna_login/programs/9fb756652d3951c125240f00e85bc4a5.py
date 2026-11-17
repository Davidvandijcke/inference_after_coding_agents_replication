#!/usr/bin/env python3
"""Score the site-adjusted treatment effect on employment at 12 months.

Usage: python score.py --data <path> --output <path>

The regression is OLS of employed_m12 on treatment and site indicators. The
reported standard error uses the HC2 heteroskedasticity-robust covariance
estimator; the p-value is from a two-sided t-test using residual degrees of
freedom.
"""

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
    outcome = data["employed_m12"].to_numpy(dtype=float)
    sites = pd.get_dummies(data["site"], drop_first=True, dtype=float).to_numpy()
    design = np.column_stack((np.ones(len(data)), treatment, sites))

    if len(data) <= design.shape[1]:
        raise ValueError("Not enough observations to estimate the model")
    xtx_inv = np.linalg.inv(design.T @ design)
    beta = xtx_inv @ design.T @ outcome
    residuals = outcome - design @ beta
    leverage = np.sum((design @ xtx_inv) * design, axis=1)
    scaled_residual_sq = residuals**2 / (1.0 - leverage)
    meat = design.T @ (design * scaled_residual_sq[:, None])
    covariance = xtx_inv @ meat @ xtx_inv

    estimate = float(beta[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    df = len(data) - design.shape[1]
    p_value = float(2.0 * t.sf(abs(estimate / std_error), df))
    return {
        "estimate": estimate,
        "std_error": std_error,
        "p_value": p_value,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Input CSV dataset")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()
    result = score(args.data)
    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2)
        output_file.write("\n")


if __name__ == "__main__":
    main()
