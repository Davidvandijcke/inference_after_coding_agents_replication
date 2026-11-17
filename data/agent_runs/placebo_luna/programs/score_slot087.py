#!/usr/bin/env python3
"""Score the baseline-adjusted, block-adjusted year 1 health_not_poor effect."""
import argparse
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

OUTCOME = "health_not_poor_y1"
BASELINE = "health_not_poor_base"


def estimate(data):
    required = {"treatment", "block", OUTCOME, BASELINE}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))

    d = data[["treatment", "block", OUTCOME, BASELINE]].dropna().copy()
    if d.empty:
        raise ValueError("No complete observations for the selected analysis")

    # OLS design: treatment, block indicators (first block as reference),
    # and the baseline value of the outcome.
    block_dummies = pd.get_dummies(d["block"].astype("category"), drop_first=True, dtype=float)
    x = np.column_stack(
        [np.ones(len(d)), d["treatment"].to_numpy(dtype=float),
         block_dummies.to_numpy(dtype=float), d[BASELINE].to_numpy(dtype=float)]
    )
    y = d[OUTCOME].to_numpy(dtype=float)
    n, k = x.shape
    if n <= k or np.linalg.matrix_rank(x) < k:
        raise ValueError("The analysis design matrix is not full rank or has too few rows")

    xtx_inv = np.linalg.inv(x.T @ x)
    beta = xtx_inv @ (x.T @ y)
    residual = y - x @ beta
    # HC2 covariance: leverage-corrected squared residuals.
    leverage = np.einsum("ij,jk,ik->i", x, xtx_inv, x)
    weighted_resid_sq = residual**2 / (1.0 - leverage)
    meat = x.T @ (x * weighted_resid_sq[:, None])
    covariance = xtx_inv @ meat @ xtx_inv

    estimate = float(beta[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    df = n - k
    p_value = float(2.0 * student_t.sf(abs(estimate / std_error), df))
    return {"estimate": estimate, "std_error": std_error, "p_value": p_value}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV with the evaluation columns")
    parser.add_argument("--output", required=True, help="Path for the JSON result")
    args = parser.parse_args()
    try:
        result = estimate(pd.read_csv(args.data))
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, allow_nan=False)
            f.write("\n")
    except Exception as exc:
        print(f"score.py: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
