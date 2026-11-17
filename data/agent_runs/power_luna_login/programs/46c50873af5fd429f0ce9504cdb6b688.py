#!/usr/bin/env python3
"""Score the site-adjusted 12-month employment treatment effect."""

import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats


def main():
    parser = argparse.ArgumentParser(
        description="Estimate the site-adjusted treatment effect on employment at 12 months."
    )
    parser.add_argument("--data", required=True, help="CSV file with the evaluation columns")
    parser.add_argument("--output", required=True, help="Path for the JSON result")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = {"site", "treatment", "employed_m12"}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    if data[list(required)].isna().any().any():
        raise ValueError("Required analysis columns contain missing values")

    # Intercept, randomized assignment, and site indicators (first site omitted).
    site_dummies = pd.get_dummies(data["site"], drop_first=True, dtype=float)
    design = np.column_stack(
        [np.ones(len(data)), data["treatment"].to_numpy(dtype=float), site_dummies.to_numpy()]
    )
    outcome = data["employed_m12"].to_numpy(dtype=float)
    n, k = design.shape
    if n <= k:
        raise ValueError("Not enough observations to estimate the model")

    xtx_inverse = np.linalg.inv(design.T @ design)
    coefficients = xtx_inverse @ design.T @ outcome
    residuals = outcome - design @ coefficients

    # HC2 sandwich covariance: each squared residual is divided by (1 - leverage).
    leverage = np.sum((design @ xtx_inverse) * design, axis=1)
    if np.any(leverage >= 1):
        raise ValueError("HC2 variance is undefined for an observation with unit leverage")
    weighted_crossproduct = design.T @ ((residuals**2 / (1 - leverage))[:, None] * design)
    covariance = xtx_inverse @ weighted_crossproduct @ xtx_inverse
    estimate = float(coefficients[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    if std_error <= 0:
        raise ValueError("Treatment coefficient has no estimated standard error")
    p_value = float(2 * stats.t.sf(abs(estimate / std_error), df=n - k))

    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(
            {"estimate": estimate, "std_error": std_error, "p_value": p_value},
            output_file,
            indent=2,
        )
        output_file.write("\n")


if __name__ == "__main__":
    main()
