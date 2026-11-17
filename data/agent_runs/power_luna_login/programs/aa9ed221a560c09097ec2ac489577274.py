#!/usr/bin/env python3
"""Site-adjusted HC2 test of the 12-month employment treatment effect."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def score(data_path):
    data = pd.read_csv(data_path)
    required = {"site", "treatment", "employed_m12"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    if data[list(required)].isna().any().any():
        raise ValueError("Required analysis columns must not contain missing values")

    sites = sorted(data["site"].unique())
    treatment = data["treatment"].to_numpy(dtype=float)
    site_values = data["site"].to_numpy()
    outcome = data["employed_m12"].to_numpy(dtype=float)
    # OLS with an intercept, treatment indicator, and site fixed effects.
    columns = [np.ones(len(data)), treatment]
    columns.extend((site_values == site).astype(float) for site in sites[1:])
    design = np.column_stack(columns)

    n, k = design.shape
    if n <= k:
        raise ValueError("Not enough observations to estimate the model")
    xtx_inverse = np.linalg.inv(design.T @ design)
    coefficients = xtx_inverse @ design.T @ outcome
    residuals = outcome - design @ coefficients
    leverage = np.sum(design * (design @ xtx_inverse), axis=1)
    adjusted_residuals = residuals / (1.0 - leverage)
    meat = design.T @ ((adjusted_residuals**2)[:, None] * design)
    covariance = xtx_inverse @ meat @ xtx_inverse

    estimate = float(coefficients[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    statistic = estimate / std_error
    p_value = float(2 * stats.t.sf(abs(statistic), df=n - k))
    return {"estimate": estimate, "std_error": std_error, "p_value": p_value}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV dataset")
    parser.add_argument("--output", required=True, help="Output JSON file")
    args = parser.parse_args()

    result = score(args.data)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2)
        output_file.write("\n")


if __name__ == "__main__":
    main()
