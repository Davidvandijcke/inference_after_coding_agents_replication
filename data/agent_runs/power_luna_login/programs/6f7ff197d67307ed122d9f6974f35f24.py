#!/usr/bin/env python3
"""Score the site-adjusted ITT effect on employment at 12 months."""

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import t


def estimate(data_path):
    data = pd.read_csv(data_path)
    required = {"site", "treatment", "employed_m12"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    # Intercept, treatment assignment, and site indicators (first site omitted).
    site_indicators = pd.get_dummies(data["site"], drop_first=True, dtype=float).to_numpy()
    design = np.column_stack(
        [np.ones(len(data)), data["treatment"].to_numpy(dtype=float), site_indicators]
    )
    outcome = data["employed_m12"].to_numpy(dtype=float)

    coefficients = np.linalg.lstsq(design, outcome, rcond=None)[0]
    residuals = outcome - design @ coefficients
    bread = np.linalg.inv(design.T @ design)
    leverage = np.sum((design @ bread) * design, axis=1)
    adjusted_residual_sq = residuals**2 / (1.0 - leverage)
    meat = design.T @ (adjusted_residual_sq[:, None] * design)
    covariance = bread @ meat @ bread

    estimate = float(coefficients[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    degrees_freedom = len(data) - design.shape[1]
    statistic = estimate / std_error
    p_value = float(2.0 * t.sf(abs(statistic), degrees_freedom))
    return {
        "estimate": estimate,
        "std_error": std_error,
        "p_value": p_value,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Estimate the site-adjusted treatment effect on employment at 12 months."
    )
    parser.add_argument("--data", required=True, help="Path to participant CSV")
    parser.add_argument("--output", required=True, help="Path for the JSON result")
    args = parser.parse_args()

    result = estimate(args.data)
    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2)
        output_file.write("\n")


if __name__ == "__main__":
    main()
