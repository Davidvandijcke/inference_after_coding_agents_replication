#!/usr/bin/env python3
"""Score the site-adjusted 12-month employment treatment effect."""

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


def estimate(data_path):
    data = pd.read_csv(data_path)
    required = {"site", "treatment", "employed_m12"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    # Match OLS with an intercept, the treatment indicator, and site indicators.
    site_indicators = pd.get_dummies(data["site"], drop_first=True, dtype=float)
    design = np.column_stack(
        [
            np.ones(len(data)),
            data["treatment"].to_numpy(dtype=float),
            site_indicators.to_numpy(dtype=float),
        ]
    )
    outcome = data["employed_m12"].to_numpy(dtype=float)
    n, k = design.shape
    if n <= k:
        raise ValueError("Not enough observations to estimate the model")

    xtx_inv = np.linalg.inv(design.T @ design)
    coefficients = xtx_inv @ design.T @ outcome
    residuals = outcome - design @ coefficients

    # HC2 sandwich covariance: each squared residual is divided by 1 - leverage.
    leverage = np.sum((design @ xtx_inv) * design, axis=1)
    if np.any(1.0 - leverage <= 0):
        raise ValueError("HC2 covariance is undefined for this dataset")
    weighted_x = design * (residuals**2 / (1.0 - leverage))[:, None]
    covariance = xtx_inv @ design.T @ weighted_x @ xtx_inv

    treatment_estimate = float(coefficients[1])
    standard_error = float(np.sqrt(covariance[1, 1]))
    statistic = treatment_estimate / standard_error
    p_value = float(2.0 * student_t.sf(abs(statistic), df=n - k))
    return {
        "estimate": treatment_estimate,
        "std_error": standard_error,
        "p_value": p_value,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Estimate the site-adjusted 12-month employment effect."
    )
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    result = estimate(args.data)
    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2)
        output_file.write("\n")


if __name__ == "__main__":
    main()
