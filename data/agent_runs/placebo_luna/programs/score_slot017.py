#!/usr/bin/env python3
"""Block-adjusted treatment effect on wtp_zero_if_ineffective."""

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import t


OUTCOME = "wtp_zero_if_ineffective"


def score(data_path):
    data = pd.read_csv(data_path)
    needed = {"treatment", "block", OUTCOME}
    missing = needed.difference(data.columns)
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))

    data = data[["treatment", "block", OUTCOME]].dropna().copy()
    if data.empty:
        raise ValueError("No complete observations for the chosen analysis")

    # Match OLS with an intercept, treatment indicator, and block indicators.
    block_dummies = pd.get_dummies(data["block"], drop_first=True, dtype=float)
    x = np.column_stack(
        [np.ones(len(data)), data["treatment"].astype(float), block_dummies.to_numpy()]
    )
    y = data[OUTCOME].astype(float).to_numpy()
    beta, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    if rank != x.shape[1]:
        raise ValueError("Design matrix is rank deficient")
    df_resid = len(y) - rank
    if df_resid <= 0:
        raise ValueError("Insufficient residual degrees of freedom")

    xtx_inv = np.linalg.inv(x.T @ x)
    residuals = y - x @ beta
    leverage = np.sum((x @ xtx_inv) * x, axis=1)
    if np.any(leverage >= 1):
        raise ValueError("HC2 variance is undefined for an observation with leverage 1")
    weighted_x = x * (residuals**2 / (1 - leverage))[:, None]
    covariance = xtx_inv @ (x.T @ weighted_x) @ xtx_inv

    estimate = float(beta[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    statistic = estimate / std_error
    p_value = float(2 * t.sf(abs(statistic), df_resid))
    return {
        "estimate": estimate,
        "std_error": std_error,
        "p_value": p_value,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="CSV input path")
    parser.add_argument("--output", required=True, help="JSON output path")
    args = parser.parse_args()
    result = score(args.data)
    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2)
        output_file.write("\n")


if __name__ == "__main__":
    main()
