#!/usr/bin/env python3
"""Score the site-adjusted treatment effect on 12-month employment."""

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


def score(data_path):
    data = pd.read_csv(data_path)
    required = {"site", "treatment", "employed_m12"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    if data[list(required)].isna().any().any():
        raise ValueError("Required analysis columns contain missing values")

    site_indicators = pd.get_dummies(data["site"], drop_first=True, dtype=float)
    treatment = data["treatment"].to_numpy(dtype=float)
    design = np.column_stack(
        [np.ones(len(data)), treatment, site_indicators.to_numpy(dtype=float)]
    )
    outcome = data["employed_m12"].to_numpy(dtype=float)
    n, k = design.shape
    if n <= k:
        raise ValueError("Not enough observations to estimate the model")

    xtx_inverse = np.linalg.inv(design.T @ design)
    coefficients = xtx_inverse @ design.T @ outcome
    residuals = outcome - design @ coefficients
    leverage = np.sum((design @ xtx_inverse) * design, axis=1)

    # HC2 sandwich covariance: residual squared divided by (1 - leverage).
    weighted_design = design * (residuals**2 / (1.0 - leverage))[:, None]
    covariance = xtx_inverse @ design.T @ weighted_design @ xtx_inverse
    estimate = float(coefficients[1])
    standard_error = float(np.sqrt(covariance[1, 1]))
    degrees_freedom = n - k
    statistic = estimate / standard_error
    p_value = float(2.0 * student_t.sf(abs(statistic), degrees_freedom))

    return {
        "estimate": estimate,
        "std_error": standard_error,
        "p_value": p_value,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Input CSV file")
    parser.add_argument("--output", required=True, help="Output JSON file")
    args = parser.parse_args()
    result = score(args.data)
    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2)
        output_file.write("\n")


if __name__ == "__main__":
    main()
