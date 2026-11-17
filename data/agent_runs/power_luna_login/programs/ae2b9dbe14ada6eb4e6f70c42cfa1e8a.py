#!/usr/bin/env python3
"""Score the prespecified lead analysis: site-adjusted 12-month employment."""

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


def score(data_path):
    data = pd.read_csv(data_path)
    required = {"employed_m12", "treatment", "site"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    # Intercept, treatment indicator, and site indicators (first site omitted).
    site_indicators = pd.get_dummies(data["site"], drop_first=True, dtype=float)
    x = np.column_stack(
        [np.ones(len(data)), data["treatment"].to_numpy(dtype=float),
         site_indicators.to_numpy(dtype=float)]
    )
    y = data["employed_m12"].to_numpy(dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Analysis columns must not contain missing or non-finite values")

    xtx_inv = np.linalg.inv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    residuals = y - x @ beta
    # HC2: each squared residual is divided by (1 - its OLS leverage).
    leverage = np.einsum("ij,jk,ik->i", x, xtx_inv, x)
    adjusted_sq_residuals = residuals**2 / (1.0 - leverage)
    meat = x.T @ (x * adjusted_sq_residuals[:, None])
    covariance = xtx_inv @ meat @ xtx_inv

    estimate = float(beta[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    df = len(y) - x.shape[1]
    statistic = estimate / std_error
    p_value = float(2.0 * student_t.sf(abs(statistic), df))
    return {
        "estimate": estimate,
        "std_error": std_error,
        "p_value": p_value,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="CSV file with evaluation data")
    parser.add_argument("--output", required=True, help="Path for JSON results")
    args = parser.parse_args()
    result = score(args.data)
    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2)
        output_file.write("\n")


if __name__ == "__main__":
    main()
