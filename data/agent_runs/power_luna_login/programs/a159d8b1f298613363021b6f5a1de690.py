#!/usr/bin/env python3
"""Score the site-adjusted treatment effect on 12-month employment."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


def main():
    parser = argparse.ArgumentParser(
        description="Estimate the treatment effect on employed_m12, adjusted for site."
    )
    parser.add_argument("--data", required=True, help="Input CSV with evaluation columns")
    parser.add_argument("--output", required=True, help="Path for the JSON result")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = {"site", "treatment", "employed_m12"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    if data[list(required)].isna().any().any():
        raise ValueError("Required analysis columns contain missing values")

    # OLS design: intercept, treatment, and site indicators (one site omitted).
    site_indicators = pd.get_dummies(data["site"], drop_first=True, dtype=float)
    design = np.column_stack(
        [np.ones(len(data)), data["treatment"].to_numpy(dtype=float), site_indicators.to_numpy()]
    )
    outcome = data["employed_m12"].to_numpy(dtype=float)
    xtx_inv = np.linalg.inv(design.T @ design)
    coefficients = xtx_inv @ design.T @ outcome
    residuals = outcome - design @ coefficients

    # HC2 sandwich covariance: scale each squared residual by 1 - leverage.
    leverage = np.einsum("ij,jk,ik->i", design, xtx_inv, design)
    meat = design.T @ ((residuals**2 / (1.0 - leverage))[:, None] * design)
    covariance = xtx_inv @ meat @ xtx_inv
    estimate = float(coefficients[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    degrees_freedom = len(data) - design.shape[1]
    statistic = estimate / std_error
    p_value = float(2.0 * student_t.sf(abs(statistic), degrees_freedom))

    result = {
        "estimate": estimate,
        "std_error": std_error,
        "p_value": p_value,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
