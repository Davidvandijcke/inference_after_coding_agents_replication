#!/usr/bin/env python3
"""Score the site-adjusted treatment effect on employment at month 12."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t


def estimate(data_path: Path) -> dict[str, float]:
    data = pd.read_csv(data_path)
    required = {"site", "treatment", "employed_m12"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    if data.empty or data[list(required)].isna().any().any():
        raise ValueError("The input must have rows and complete analysis columns.")

    # OLS design: intercept, treatment indicator, and site indicators.
    site_indicators = pd.get_dummies(data["site"], drop_first=True, dtype=float)
    design = np.column_stack(
        [np.ones(len(data)), data["treatment"].to_numpy(dtype=float),
         site_indicators.to_numpy(dtype=float)]
    )
    outcome = data["employed_m12"].to_numpy(dtype=float)

    xtx_inverse = np.linalg.inv(design.T @ design)
    coefficients = xtx_inverse @ design.T @ outcome
    residuals = outcome - design @ coefficients
    leverage = np.sum((design @ xtx_inverse) * design, axis=1)

    # HC2 sandwich covariance, with residual degrees of freedom for the t test.
    weighted_design = design * (residuals**2 / (1.0 - leverage))[:, None]
    meat = design.T @ weighted_design
    covariance = xtx_inverse @ meat @ xtx_inverse
    standard_error = float(np.sqrt(covariance[1, 1]))
    estimate_value = float(coefficients[1])
    degrees_freedom = len(data) - design.shape[1]
    statistic = estimate_value / standard_error
    p_value = float(2.0 * t.sf(abs(statistic), degrees_freedom))

    return {
        "estimate": estimate_value,
        "std_error": standard_error,
        "p_value": p_value,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate the site-adjusted effect on employment at 12 months."
    )
    parser.add_argument("--data", required=True, type=Path, help="Input CSV path")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON path")
    args = parser.parse_args()
    result = estimate(args.data)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
