#!/usr/bin/env python3
"""Estimate the treatment effect on reported happiness immediately post-video.

Usage: python score.py --data <path> --output <path>
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t


OUTCOME = "felt_happy_post"


def score(data_path: Path) -> dict[str, float]:
    data = pd.read_csv(data_path)
    required = {"treatment", "block", OUTCOME}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    sample = data[["treatment", "block", OUTCOME]].dropna()
    if sample.empty:
        raise ValueError(f"No complete observations for {OUTCOME}")

    # OLS with an intercept, treatment indicator, and block fixed effects.
    blocks = pd.get_dummies(sample["block"].astype("category"), drop_first=True, dtype=float)
    x = np.column_stack(
        [np.ones(len(sample)), sample["treatment"].to_numpy(dtype=float), blocks.to_numpy()]
    )
    y = sample[OUTCOME].to_numpy(dtype=float)
    xtx_inv = np.linalg.pinv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    residuals = y - x @ beta
    leverage = np.einsum("ij,jk,ik->i", x, xtx_inv, x)
    rank = np.linalg.matrix_rank(x)
    df = len(y) - rank
    if df <= 0 or np.any(1.0 - leverage <= 0):
        raise ValueError("Insufficient observations for HC2 inference")

    # HC2 sandwich covariance: (X'X)^-1 X' diag(e_i^2/(1-h_i)) X (X'X)^-1.
    weighted_x = x * (residuals**2 / (1.0 - leverage))[:, None]
    covariance = xtx_inv @ (x.T @ weighted_x) @ xtx_inv
    estimate = float(beta[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    statistic = estimate / std_error
    p_value = float(2.0 * t.sf(abs(statistic), df))
    return {"estimate": estimate, "std_error": std_error, "p_value": p_value}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="CSV dataset")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON path")
    args = parser.parse_args()

    result = score(args.data)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
