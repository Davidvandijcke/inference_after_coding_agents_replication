#!/usr/bin/env python3
"""Score the site-adjusted treatment effect on employment at 12 months."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


def estimate(data: pd.DataFrame) -> dict[str, float]:
    """OLS employment_m12 ~ treatment + site indicators, with HC2 inference."""
    required = {"site", "treatment", "employed_m12"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    frame = data[["site", "treatment", "employed_m12"]].dropna()
    y = frame["employed_m12"].to_numpy(dtype=float)
    treatment = frame["treatment"].to_numpy(dtype=float)
    sites = pd.get_dummies(frame["site"], drop_first=True, dtype=float).to_numpy()
    X = np.column_stack((np.ones(len(frame)), treatment, sites))
    n, k = X.shape
    if n <= k:
        raise ValueError("Insufficient complete observations to estimate the model")

    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    residuals = y - X @ beta
    leverage = np.sum((X @ xtx_inv) * X, axis=1)
    adjusted_sq_residuals = residuals**2 / (1.0 - leverage)
    meat = X.T @ (X * adjusted_sq_residuals[:, None])
    covariance = xtx_inv @ meat @ xtx_inv

    estimate_value = float(beta[1])
    standard_error = float(np.sqrt(covariance[1, 1]))
    statistic = estimate_value / standard_error
    p_value = float(2.0 * student_t.sf(abs(statistic), df=n - k))
    return {
        "estimate": estimate_value,
        "std_error": standard_error,
        "p_value": p_value,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate the site-adjusted program effect on employment at 12 months."
    )
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    result = estimate(pd.read_csv(args.data))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
