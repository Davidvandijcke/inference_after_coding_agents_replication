#!/usr/bin/env python3
"""Score the Pathways to Work 12-month employment treatment effect.

Usage: python score.py --data <path> --output <path>

Fits OLS of employed_m12 on treatment and site indicators, then reports the
treatment coefficient, its HC2 standard error, and a two-sided t-test p-value.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


def estimate_effect(data: pd.DataFrame) -> dict[str, float]:
    required = {"site", "treatment", "employed_m12"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    if data[list(required)].isna().any().any():
        raise ValueError("Required analysis columns contain missing values")

    treatment = data["treatment"].to_numpy(dtype=float)
    outcome = data["employed_m12"].to_numpy(dtype=float)
    sites = pd.get_dummies(data["site"], drop_first=True, dtype=float).to_numpy()
    design = np.column_stack((np.ones(len(data)), treatment, sites))

    n, k = design.shape
    if n <= k:
        raise ValueError("Not enough observations to estimate the model")
    xtx_inverse = np.linalg.pinv(design.T @ design)
    coefficients = xtx_inverse @ design.T @ outcome
    residuals = outcome - design @ coefficients
    leverage = np.sum((design @ xtx_inverse) * design, axis=1)

    # HC2 sandwich covariance: residual contributions are divided by 1 - h_ii.
    meat = design.T @ ((residuals**2 / (1.0 - leverage))[:, None] * design)
    covariance = xtx_inverse @ meat @ xtx_inverse
    estimate = float(coefficients[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    degrees_of_freedom = n - k
    statistic = estimate / std_error
    p_value = float(2.0 * student_t.sf(abs(statistic), degrees_of_freedom))

    return {
        "estimate": estimate,
        "std_error": std_error,
        "p_value": p_value,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    result = estimate_effect(pd.read_csv(args.data))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
