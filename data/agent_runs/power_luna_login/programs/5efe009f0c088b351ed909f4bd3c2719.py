#!/usr/bin/env python3
"""Score the Pathways to Work 12-month employment effect.

Usage: python score.py --data <path> --output <path>

Fits employed_m12 on treatment and site indicators, using the HC2
heteroskedasticity-robust covariance estimator and a two-sided t test.
"""

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


def estimate(data_path):
    data = pd.read_csv(data_path)
    required = {"employed_m12", "treatment", "site"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    if data[list(required)].isna().any().any():
        raise ValueError("Required analysis columns contain missing values")

    # The first site is the reference category; the intercept and remaining
    # site indicators implement site fixed effects.
    site = pd.get_dummies(data["site"].astype("category"), drop_first=True,
                          dtype=float).to_numpy()
    design = np.column_stack((np.ones(len(data)),
                              data["treatment"].to_numpy(dtype=float), site))
    outcome = data["employed_m12"].to_numpy(dtype=float)
    if len(data) <= design.shape[1]:
        raise ValueError("Not enough observations to estimate the model")

    xtx_inverse = np.linalg.inv(design.T @ design)
    coefficients = xtx_inverse @ design.T @ outcome
    residuals = outcome - design @ coefficients
    leverage = np.sum((design @ xtx_inverse) * design, axis=1)
    hc2_meat = (design * (residuals**2 / (1.0 - leverage))[:, None]).T @ design
    covariance = xtx_inverse @ hc2_meat @ xtx_inverse

    estimate = float(coefficients[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    degrees_freedom = len(data) - design.shape[1]
    statistic = estimate / std_error
    p_value = float(2.0 * student_t.sf(abs(statistic), degrees_freedom))
    return {"estimate": estimate, "std_error": std_error, "p_value": p_value}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV file")
    parser.add_argument("--output", required=True, help="Output JSON file")
    args = parser.parse_args()
    result = estimate(args.data)
    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2)
        output_file.write("\n")


if __name__ == "__main__":
    main()
