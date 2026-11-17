#!/usr/bin/env python3
"""Score the site-adjusted treatment effect on 12-month employment."""

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import t


def score(data_path):
    data = pd.read_csv(data_path)
    required = {"site", "treatment", "employed_m12"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    outcome = data["employed_m12"].to_numpy(dtype=float)
    treatment = data["treatment"].to_numpy(dtype=float)
    site_indicators = pd.get_dummies(
        data["site"], drop_first=True, dtype=float
    ).to_numpy()
    design = np.column_stack(
        (np.ones(len(data)), treatment, site_indicators)
    )

    n, n_parameters = design.shape
    if n <= n_parameters:
        raise ValueError("Not enough rows to estimate the model")
    if not np.isfinite(outcome).all() or not np.isfinite(design).all():
        raise ValueError("Required analysis columns must be complete and finite")

    xtx_inverse = np.linalg.inv(design.T @ design)
    coefficients = xtx_inverse @ design.T @ outcome
    residuals = outcome - design @ coefficients
    leverage = np.sum((design @ xtx_inverse) * design, axis=1)
    meat = design.T @ ((residuals**2 / (1.0 - leverage))[:, None] * design)
    covariance = xtx_inverse @ meat @ xtx_inverse

    estimate = float(coefficients[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    degrees_freedom = n - n_parameters
    statistic = estimate / std_error
    p_value = float(2.0 * t.sf(abs(statistic), degrees_freedom))
    return {
        "estimate": estimate,
        "std_error": std_error,
        "p_value": p_value,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Estimate the site-adjusted effect on 12-month employment."
    )
    parser.add_argument("--data", required=True, help="Input CSV dataset")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    result = score(args.data)
    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2)
        output_file.write("\n")


if __name__ == "__main__":
    main()
