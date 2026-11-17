#!/usr/bin/env python3
"""Score the 12-month employment effect among participants unemployed at baseline."""

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import t


def estimate(data_path):
    data = pd.read_csv(data_path)
    subgroup = data.loc[data["employed_baseline"] == 0].copy()

    if subgroup.empty:
        raise ValueError("No participants were unemployed at baseline.")

    sites = sorted(subgroup["site"].unique())
    treatment = subgroup["treatment"].to_numpy(dtype=float)
    site_values = subgroup["site"].to_numpy()

    # Intercept, treatment, and site indicators (first site is the reference).
    columns = [np.ones(len(subgroup)), treatment]
    columns.extend((site_values == site).astype(float) for site in sites[1:])
    design = np.column_stack(columns)
    outcome = subgroup["employed_m12"].to_numpy(dtype=float)

    bread = np.linalg.inv(design.T @ design)
    coefficients = bread @ design.T @ outcome
    residuals = outcome - design @ coefficients
    leverage = np.sum((design @ bread) * design, axis=1)

    # HC2 sandwich covariance estimator.
    weighted_design = design * (residuals**2 / (1.0 - leverage))[:, None]
    covariance = bread @ (design.T @ weighted_design) @ bread

    estimate = float(coefficients[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    degrees_freedom = len(subgroup) - design.shape[1]
    p_value = float(2.0 * t.sf(abs(estimate / std_error), degrees_freedom))
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

    result = estimate(args.data)
    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2)
        output_file.write("\n")


if __name__ == "__main__":
    main()
