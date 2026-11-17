#!/usr/bin/env python3
"""Score the Pathways to Work 12-month employment treatment effect."""

import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats


def estimate(data_path):
    data = pd.read_csv(data_path)
    required = {"site", "treatment", "employed_m12"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    if data[["site", "treatment", "employed_m12"]].isna().any().any():
        raise ValueError("Required analysis columns must not contain missing values")

    # OLS with an intercept, treatment assignment, and site fixed effects.
    treatment = data["treatment"].to_numpy(dtype=float)
    site_effects = pd.get_dummies(data["site"], drop_first=True, dtype=float).to_numpy()
    design = np.column_stack((np.ones(len(data)), treatment, site_effects))
    outcome = data["employed_m12"].to_numpy(dtype=float)

    xtx_inverse = np.linalg.inv(design.T @ design)
    coefficients = xtx_inverse @ design.T @ outcome
    residuals = outcome - design @ coefficients
    leverage = np.sum((design @ xtx_inverse) * design, axis=1)

    # HC2 sandwich covariance estimator.
    weighted_crossproducts = design.T @ (
        (residuals**2 / (1.0 - leverage))[:, None] * design
    )
    covariance = xtx_inverse @ weighted_crossproducts @ xtx_inverse
    standard_error = float(np.sqrt(covariance[1, 1]))
    degrees_freedom = len(data) - design.shape[1]
    statistic = float(coefficients[1] / standard_error)
    p_value = float(2.0 * stats.t.sf(abs(statistic), degrees_freedom))

    return {
        "estimate": float(coefficients[1]),
        "std_error": standard_error,
        "p_value": p_value,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV dataset")
    parser.add_argument("--output", required=True, help="Path for JSON result")
    args = parser.parse_args()

    result = estimate(args.data)
    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2)
        output_file.write("\n")


if __name__ == "__main__":
    main()
