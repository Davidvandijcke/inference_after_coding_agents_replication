#!/usr/bin/env python3
"""Score the site- and baseline-employment-adjusted 12-month employment effect."""

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="CSV file with evaluation observations")
    parser.add_argument("--output", required=True, help="Path for the JSON result")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = ["site", "treatment", "employed_baseline", "employed_m12"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    if data[required].isna().any().any():
        raise ValueError("Required analysis columns contain missing values")

    treatment = data["treatment"].to_numpy(dtype=float)
    baseline_employment = data["employed_baseline"].to_numpy(dtype=float)
    outcome = data["employed_m12"].to_numpy(dtype=float)
    site_indicators = pd.get_dummies(data["site"], drop_first=True, dtype=float).to_numpy()

    # OLS: outcome ~ treatment + site indicators + baseline employment.
    design = np.column_stack(
        [np.ones(len(data)), treatment, baseline_employment, site_indicators]
    )
    n, k = design.shape
    if n <= k:
        raise ValueError("Not enough observations to estimate the model")

    bread = np.linalg.inv(design.T @ design)
    coefficients = bread @ design.T @ outcome
    residuals = outcome - design @ coefficients
    leverage = np.sum((design @ bread) * design, axis=1)

    # HC2 sandwich covariance: each squared residual is divided by (1 - h_ii).
    adjusted_squares = residuals**2 / (1.0 - leverage)
    covariance = bread @ (design.T @ (adjusted_squares[:, None] * design)) @ bread
    estimate = float(coefficients[1])
    standard_error = float(np.sqrt(covariance[1, 1]))
    statistic = estimate / standard_error
    p_value = float(2.0 * student_t.sf(abs(statistic), df=n - k))

    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(
            {
                "estimate": estimate,
                "std_error": standard_error,
                "p_value": p_value,
            },
            output_file,
            indent=2,
        )
        output_file.write("\n")


if __name__ == "__main__":
    main()
