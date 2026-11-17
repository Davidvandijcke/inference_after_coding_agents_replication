#!/usr/bin/env python3
"""Score the site-adjusted intention-to-treat effect on 12-month employment.

Usage: python score.py --data <path> --output <path>

Fits OLS: employed_m12 ~ treatment + site indicators. The treatment estimate
uses HC2 heteroskedasticity-robust standard errors and a two-sided t-test.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


def score(data_path: str) -> dict[str, float]:
    data = pd.read_csv(data_path)
    required = {"site", "treatment", "employed_m12"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    if data[list(required)].isna().any().any():
        raise ValueError("Required analysis columns contain missing values")

    treatment = data["treatment"].to_numpy(dtype=float)
    # Use one indicator per site except the reference site.
    site_dummies = pd.get_dummies(data["site"], drop_first=True, dtype=float).to_numpy()
    design = np.column_stack((np.ones(len(data)), treatment, site_dummies))
    outcome = data["employed_m12"].to_numpy(dtype=float)

    n, k = design.shape
    if n <= k:
        raise ValueError("Not enough observations to estimate the model")
    xtx_inverse = np.linalg.inv(design.T @ design)
    coefficients = xtx_inverse @ design.T @ outcome
    residuals = outcome - design @ coefficients
    leverage = np.sum((design @ xtx_inverse) * design, axis=1)

    # HC2 sandwich covariance: (X'X)^-1 X' diag(e_i^2/(1-h_i)) X (X'X)^-1
    weighted_design = design * (residuals**2 / (1.0 - leverage))[:, None]
    covariance = xtx_inverse @ design.T @ weighted_design @ xtx_inverse

    estimate = float(coefficients[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    degrees_freedom = n - k
    statistic = estimate / std_error
    p_value = float(2.0 * student_t.sf(abs(statistic), degrees_freedom))
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

    result = score(args.data)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
