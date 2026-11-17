#!/usr/bin/env python3
"""Score the preselected year-2 current-smoking treatment effect.

Usage: python score.py --data <path> --output <path>

Fits OLS of current_smoker_y2 on treatment, block indicators, and
current_smoker_base among observations with those fields recorded. Reports the
treatment coefficient, HC2 standard error, and two-sided t-test p-value.
"""

import argparse
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


def estimate(path):
    data = pd.read_csv(path)
    required = ["current_smoker_y2", "treatment", "block", "current_smoker_base"]
    missing = [name for name in required if name not in data.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    sample = data[required].dropna().copy()
    if sample.empty or sample["treatment"].nunique() != 2:
        raise ValueError("Complete cases must include both treatment assignments")

    # Match block fixed effects with an intercept and one omitted reference block.
    blocks = pd.get_dummies(sample["block"].astype("category"), drop_first=True, dtype=float)
    design = np.column_stack(
        [
            np.ones(len(sample)),
            sample["treatment"].to_numpy(dtype=float),
            blocks.to_numpy(dtype=float),
            sample["current_smoker_base"].to_numpy(dtype=float),
        ]
    )
    outcome = sample["current_smoker_y2"].to_numpy(dtype=float)
    n, p = design.shape
    if n <= p or np.linalg.matrix_rank(design) != p:
        raise ValueError("Analysis design is not full rank or has no residual degrees of freedom")

    xtx_inv = np.linalg.inv(design.T @ design)
    coef = xtx_inv @ design.T @ outcome
    residual = outcome - design @ coef
    leverage = np.einsum("ij,jk,ik->i", design, xtx_inv, design)
    if np.any(1.0 - leverage <= 0):
        raise ValueError("HC2 variance is undefined for this analysis design")
    weighted = design * (residual**2 / (1.0 - leverage))[:, None]
    covariance = xtx_inv @ (design.T @ weighted) @ xtx_inv

    estimate_value = float(coef[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    statistic = estimate_value / std_error
    degrees_freedom = n - p
    p_value = float(2.0 * student_t.sf(abs(statistic), degrees_freedom))
    return {
        "estimate": estimate_value,
        "std_error": std_error,
        "p_value": p_value,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()
    try:
        result = estimate(args.data)
        with open(args.output, "w", encoding="utf-8") as output_file:
            json.dump(result, output_file, indent=2)
            output_file.write("\n")
    except Exception as error:
        print(f"score.py: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
