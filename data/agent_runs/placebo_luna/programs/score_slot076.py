#!/usr/bin/env python3
"""Score the selected block-adjusted treatment effect on follow-up career rank."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

OUTCOME = "rank_career_fu"


def score(data_path: str) -> dict[str, float]:
    data = pd.read_csv(data_path)
    required = {"treatment", "block", OUTCOME}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    sample = data[["treatment", "block", OUTCOME]].dropna().copy()
    if sample.empty:
        raise ValueError(f"No complete observations for {OUTCOME}")

    # OLS design: intercept, treatment indicator, and block fixed effects.
    block_indicators = pd.get_dummies(sample["block"].astype(str), drop_first=True, dtype=float)
    x = np.column_stack(
        [np.ones(len(sample)), sample["treatment"].to_numpy(dtype=float), block_indicators.to_numpy()]
    )
    y = sample[OUTCOME].to_numpy(dtype=float)
    n, k = x.shape
    if n <= k or np.linalg.matrix_rank(x) < k:
        raise ValueError("Insufficient observations or rank deficient treatment/block design")

    xtx_inv = np.linalg.inv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    residuals = y - x @ beta
    leverage = np.einsum("ij,jk,ik->i", x, xtx_inv, x)
    # HC2 sandwich covariance, with the codebook's residual degrees of freedom
    # used for the two-sided t test.
    adjusted_sq_resid = residuals**2 / (1.0 - leverage)
    meat = x.T @ (x * adjusted_sq_resid[:, None])
    covariance = xtx_inv @ meat @ xtx_inv
    estimate = float(beta[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    df = n - k
    statistic = estimate / std_error
    p_value = float(2.0 * student_t.sf(abs(statistic), df))

    return {"estimate": estimate, "std_error": std_error, "p_value": p_value}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="CSV dataset with the evaluation columns")
    parser.add_argument("--output", required=True, help="Path for the JSON result")
    args = parser.parse_args()

    result = score(args.data)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
