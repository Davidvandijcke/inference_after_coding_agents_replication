#!/usr/bin/env python3
"""Estimate the Pathways to Work effect on 12-month employment.

Usage: python score.py --data <path> --output <path>

Fits OLS of employed_m12 on treatment and site indicators and reports the
treatment coefficient, HC2 standard error, and two-sided t-test p-value.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


def estimate(data: pd.DataFrame) -> dict[str, float]:
    required = {"employed_m12", "treatment", "site"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    frame = data[["employed_m12", "treatment", "site"]].dropna()
    if frame.empty:
        raise ValueError("No complete observations for the chosen analysis")

    # Intercept, randomized treatment indicator, and site fixed effects.
    sites = pd.get_dummies(frame["site"], drop_first=True, dtype=float)
    x = np.column_stack(
        [np.ones(len(frame)), frame["treatment"].to_numpy(dtype=float), sites.to_numpy()]
    )
    y = frame["employed_m12"].to_numpy(dtype=float)
    n, k = x.shape
    if np.linalg.matrix_rank(x) != k or n <= k:
        raise ValueError("The analysis design matrix is not full rank")

    xtx_inv = np.linalg.inv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    residuals = y - x @ beta
    # HC2: scale each squared residual by 1 - its OLS leverage.
    leverage = np.einsum("ij,jk,ik->i", x, xtx_inv, x)
    if np.any(leverage >= 1):
        raise ValueError("HC2 covariance is undefined for an observation with leverage 1")
    weighted_x = x * (residuals**2 / (1.0 - leverage))[:, None]
    covariance = xtx_inv @ (x.T @ weighted_x) @ xtx_inv
    treatment_se = float(np.sqrt(covariance[1, 1]))
    estimate = float(beta[1])
    df_resid = n - k
    statistic = estimate / treatment_se
    p_value = float(2.0 * student_t.sf(abs(statistic), df=df_resid))

    return {
        "estimate": estimate,
        "std_error": treatment_se,
        "p_value": p_value,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV with evaluation data")
    parser.add_argument("--output", required=True, help="Path for JSON result")
    args = parser.parse_args()

    result = estimate(pd.read_csv(args.data))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
