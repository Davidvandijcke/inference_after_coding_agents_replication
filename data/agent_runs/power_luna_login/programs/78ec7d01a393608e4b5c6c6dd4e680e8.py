#!/usr/bin/env python3
"""Score the site-adjusted 12-month employment effect in a Pathways dataset."""

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


def estimate_effect(data_path):
    data = pd.read_csv(data_path)
    required = {"site", "treatment", "employed_m12"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    if data[list(required)].isna().any().any():
        raise ValueError("Required analysis columns must not contain missing values")

    # OLS design: intercept, assignment indicator, and site indicators.
    site_indicators = pd.get_dummies(data["site"], drop_first=True, dtype=float)
    x = np.column_stack(
        [
            np.ones(len(data)),
            data["treatment"].to_numpy(dtype=float),
            site_indicators.to_numpy(dtype=float),
        ]
    )
    y = data["employed_m12"].to_numpy(dtype=float)

    xtx_inverse = np.linalg.inv(x.T @ x)
    coefficients = xtx_inverse @ x.T @ y
    residuals = y - x @ coefficients
    leverage = np.sum((x @ xtx_inverse) * x, axis=1)

    # HC2 sandwich covariance: each squared residual is divided by (1 - h_ii).
    weighted_x = x * (residuals**2 / (1.0 - leverage))[:, None]
    meat = x.T @ weighted_x
    covariance = xtx_inverse @ meat @ xtx_inverse

    estimate = float(coefficients[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    degrees_freedom = len(data) - x.shape[1]
    statistic = estimate / std_error
    p_value = float(2.0 * student_t.sf(abs(statistic), degrees_freedom))
    return {
        "estimate": estimate,
        "std_error": std_error,
        "p_value": p_value,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    result = estimate_effect(args.data)
    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2)
        output_file.write("\n")


if __name__ == "__main__":
    main()
