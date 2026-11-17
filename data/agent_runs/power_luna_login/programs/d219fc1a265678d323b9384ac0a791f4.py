#!/usr/bin/env python3
"""Score the site-adjusted 12-month employment effect in a Pathways dataset."""

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

    outcome = data["employed_m12"].to_numpy(dtype=float)
    treatment = data["treatment"].to_numpy(dtype=float)
    sites = data["site"]
    site_values = sorted(sites.unique())

    # OLS with an intercept, treatment, and site fixed effects.
    columns = [np.ones(len(data)), treatment]
    columns.extend((sites.to_numpy() == site).astype(float) for site in site_values[1:])
    design = np.column_stack(columns)
    xtx_inverse = np.linalg.inv(design.T @ design)
    coefficients = xtx_inverse @ (design.T @ outcome)
    residuals = outcome - design @ coefficients

    # HC2 sandwich covariance and a two-sided t test using residual degrees of freedom.
    leverage = np.einsum("ij,jk,ik->i", design, xtx_inverse, design)
    adjusted_residual_sq = residuals**2 / (1.0 - leverage)
    meat = design.T @ (adjusted_residual_sq[:, None] * design)
    covariance = xtx_inverse @ meat @ xtx_inverse
    estimate = float(coefficients[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    degrees_freedom = len(data) - design.shape[1]
    p_value = float(2.0 * student_t.sf(abs(estimate / std_error), degrees_freedom))

    return {
        "estimate": estimate,
        "std_error": std_error,
        "p_value": p_value,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="CSV dataset with the Pathways columns")
    parser.add_argument("--output", required=True, help="Path for the JSON result")
    args = parser.parse_args()

    result = score(args.data)
    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2)
        output_file.write("\n")


if __name__ == "__main__":
    main()
