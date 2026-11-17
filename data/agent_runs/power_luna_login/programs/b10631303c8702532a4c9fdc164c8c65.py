#!/usr/bin/env python3
"""Score the 12-month employment effect with site-adjusted OLS and HC2 SEs."""

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
        raise ValueError("Analysis columns must not contain missing values")

    treatment = data["treatment"].to_numpy(dtype=float)
    outcome = data["employed_m12"].to_numpy(dtype=float)
    site_indicators = pd.get_dummies(
        data["site"], drop_first=True, dtype=float
    ).to_numpy()
    design = np.column_stack((np.ones(len(data)), treatment, site_indicators))

    coefficients, _, rank, _ = np.linalg.lstsq(design, outcome, rcond=None)
    residuals = outcome - design @ coefficients
    bread = np.linalg.pinv(design.T @ design)
    leverage = np.einsum("ij,jk,ik->i", design, bread, design)
    if np.any(leverage >= 1):
        raise ValueError("HC2 variance is undefined for an observation with leverage 1")
    weighted_cross_product = design.T @ (
        (residuals**2 / (1 - leverage))[:, None] * design
    )
    covariance = bread @ weighted_cross_product @ bread

    estimate = float(coefficients[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    residual_df = len(data) - rank
    if residual_df <= 0 or std_error <= 0:
        raise ValueError("Insufficient data to estimate the treatment effect")
    statistic = estimate / std_error
    p_value = float(2 * student_t.sf(abs(statistic), residual_df))
    return {
        "estimate": estimate,
        "std_error": std_error,
        "p_value": p_value,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    result = score(args.data)
    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2)
        output_file.write("\n")


if __name__ == "__main__":
    main()
