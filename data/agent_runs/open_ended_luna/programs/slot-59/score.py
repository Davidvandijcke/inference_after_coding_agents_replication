#!/usr/bin/env python3
"""Score the site-adjusted ITT effect on training enrollment at month 6."""

import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats


def estimate(data):
    required = {"site", "treatment", "training_m6"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    if data[list(required)].isna().any().any():
        raise ValueError("Required analysis columns contain missing values")

    outcome = data["training_m6"].to_numpy(dtype=float)
    treatment = data["treatment"].to_numpy(dtype=float)
    site_indicators = pd.get_dummies(
        data["site"].astype(str), drop_first=True, dtype=float
    ).to_numpy()
    design = np.column_stack(
        [np.ones(len(data)), treatment, site_indicators]
    )

    n, k = design.shape
    if n <= k:
        raise ValueError("Not enough observations to estimate the model")

    xtx_inv = np.linalg.inv(design.T @ design)
    coefficients = xtx_inv @ design.T @ outcome
    residuals = outcome - design @ coefficients
    leverage = np.sum((design @ xtx_inv) * design, axis=1)
    hc2_meat = design.T @ (
        (residuals**2 / (1.0 - leverage))[:, None] * design
    )
    covariance = xtx_inv @ hc2_meat @ xtx_inv

    effect = float(coefficients[1])
    standard_error = float(np.sqrt(covariance[1, 1]))
    statistic = effect / standard_error
    p_value = float(2.0 * stats.t.sf(abs(statistic), df=n - k))
    return {
        "estimate": effect,
        "std_error": standard_error,
        "p_value": p_value,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Input CSV dataset")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    result = estimate(pd.read_csv(args.data))
    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(result, output_file)
        output_file.write("\n")


if __name__ == "__main__":
    main()
