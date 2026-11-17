#!/usr/bin/env python3
"""Score the site-adjusted ITT effect on 12-month employment.

Usage: python score.py --data <path> --output <path>

Fits OLS of employed_m12 on treatment and site indicators, with an intercept.
The treatment test uses the HC2 robust standard error and a two-sided t test
with the residual degrees of freedom, matching the supplied codebook.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def score(data_path: Path) -> dict[str, float]:
    data = pd.read_csv(data_path)
    required = {"site", "treatment", "employed_m12"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    if data.empty or data[list(required)].isna().any().any():
        raise ValueError("Required analysis columns must contain observations")

    site = pd.get_dummies(data["site"], drop_first=True, dtype=float).to_numpy()
    treatment = data["treatment"].to_numpy(dtype=float)
    outcome = data["employed_m12"].to_numpy(dtype=float)
    design = np.column_stack((np.ones(len(data)), site, treatment))
    n, k = design.shape
    if n <= k:
        raise ValueError("Not enough observations to estimate the model")

    xtx_inverse = np.linalg.inv(design.T @ design)
    coefficients = xtx_inverse @ design.T @ outcome
    residuals = outcome - design @ coefficients
    leverage = np.einsum("ij,jk,ik->i", design, xtx_inverse, design)
    if np.any(1.0 - leverage <= 0):
        raise ValueError("HC2 variance is undefined for a leverage-one observation")

    scaled_scores = design * (residuals**2 / (1.0 - leverage))[:, None]
    covariance = xtx_inverse @ (design.T @ scaled_scores) @ xtx_inverse
    treatment_index = design.shape[1] - 1
    estimate = float(coefficients[treatment_index])
    std_error = float(np.sqrt(covariance[treatment_index, treatment_index]))
    if not np.isfinite(std_error) or std_error <= 0:
        raise ValueError("Could not calculate a positive finite standard error")
    statistic = estimate / std_error
    p_value = float(2.0 * stats.t.sf(abs(statistic), df=n - k))
    return {"estimate": estimate, "std_error": std_error, "p_value": p_value}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Input CSV path")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON path")
    args = parser.parse_args()

    result = score(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
