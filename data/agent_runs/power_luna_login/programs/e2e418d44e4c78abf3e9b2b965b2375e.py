#!/usr/bin/env python3
"""Score the 12-month employment treatment effect with site fixed effects."""

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
        raise ValueError("Required analysis columns must not contain missing values")

    # Intercept, treatment assignment, and site indicators (one reference site).
    site_indicators = pd.get_dummies(data["site"], drop_first=True, dtype=float)
    design = np.column_stack(
        [
            np.ones(len(data), dtype=float),
            data["treatment"].to_numpy(dtype=float),
            site_indicators.to_numpy(dtype=float),
        ]
    )
    outcome = data["employed_m12"].to_numpy(dtype=float)
    n, k = design.shape
    if n <= k or np.linalg.matrix_rank(design) != k:
        raise ValueError("The design matrix must have full column rank and positive residual df")

    xtx_inv = np.linalg.inv(design.T @ design)
    coefficients = xtx_inv @ design.T @ outcome
    residuals = outcome - design @ coefficients
    leverage = np.sum((design @ xtx_inv) * design, axis=1)
    hc2_meat = design.T @ ((residuals**2 / (1.0 - leverage))[:, None] * design)
    covariance = xtx_inv @ hc2_meat @ xtx_inv

    estimate = float(coefficients[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    statistic = estimate / std_error
    p_value = float(2.0 * student_t.sf(abs(statistic), df=n - k))
    return {"estimate": estimate, "std_error": std_error, "p_value": p_value}


def main():
    parser = argparse.ArgumentParser(
        description="Estimate the site-adjusted 12-month employment effect."
    )
    parser.add_argument("--data", required=True, help="Input CSV dataset")
    parser.add_argument("--output", required=True, help="Path for JSON results")
    args = parser.parse_args()
    result = score(args.data)
    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2)
        output_file.write("\n")


if __name__ == "__main__":
    main()
