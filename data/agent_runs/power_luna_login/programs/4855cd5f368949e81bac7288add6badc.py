#!/usr/bin/env python3
"""Estimate the site-adjusted ITT effect on employment at 12 months."""

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import t


def estimate_effect(data_path):
    data = pd.read_csv(data_path)
    required = {"site", "treatment", "employed_m12"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    if data[list(required)].isna().any().any():
        raise ValueError("Required analysis columns contain missing values")

    # Intercept, treatment, and site indicators implement the codebook model.
    site_indicators = pd.get_dummies(data["site"], drop_first=True, dtype=float)
    design = np.column_stack(
        [np.ones(len(data)), data["treatment"].to_numpy(dtype=float),
         site_indicators.to_numpy(dtype=float)]
    )
    outcome = data["employed_m12"].to_numpy(dtype=float)
    rank = np.linalg.matrix_rank(design)
    if len(data) <= rank:
        raise ValueError("Not enough observations to estimate residual variance")

    xtx_inverse = np.linalg.pinv(design.T @ design)
    coefficients = xtx_inverse @ design.T @ outcome
    residuals = outcome - design @ coefficients
    leverage = np.sum((design @ xtx_inverse) * design, axis=1)
    if np.any(leverage >= 1):
        raise ValueError("HC2 variance is undefined for this design")
    weighted_crossproduct = design.T @ (
        (residuals**2 / (1.0 - leverage))[:, None] * design
    )
    covariance = xtx_inverse @ weighted_crossproduct @ xtx_inverse

    estimate = float(coefficients[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    if std_error == 0:
        p_value = 0.0 if estimate != 0 else 1.0
    else:
        degrees_freedom = len(data) - rank
        p_value = float(2 * t.sf(abs(estimate / std_error), degrees_freedom))
    return {"estimate": estimate, "std_error": std_error, "p_value": p_value}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()
    result = estimate_effect(args.data)
    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2)
        output_file.write("\n")


if __name__ == "__main__":
    main()
