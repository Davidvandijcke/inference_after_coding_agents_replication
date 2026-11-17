#!/usr/bin/env python3
"""Score the blocked randomized treatment effect on year-2 job-search likelihood."""

import argparse
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


OUTCOME = "job_search_very_likely_y2"


def estimate(data_path):
    data = pd.read_csv(data_path)
    needed = [OUTCOME, "treatment", "block"]
    missing_columns = [name for name in needed if name not in data.columns]
    if missing_columns:
        raise ValueError("Missing required columns: " + ", ".join(missing_columns))

    sample = data[needed].dropna()
    y = sample[OUTCOME].to_numpy(dtype=float)
    treatment = sample["treatment"].to_numpy(dtype=float)
    blocks = pd.get_dummies(sample["block"], drop_first=True, dtype=float).to_numpy()
    # Match OLS with an intercept, treatment indicator, and block fixed effects.
    design = np.column_stack((np.ones(len(sample)), treatment, blocks))

    if len(sample) == 0 or len(np.unique(treatment)) != 2:
        raise ValueError("Complete cases must include both treatment arms")
    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("The analysis design matrix is rank deficient")

    xtx_inv = np.linalg.inv(design.T @ design)
    coefficients = xtx_inv @ design.T @ y
    residuals = y - design @ coefficients
    leverage = np.einsum("ij,jk,ik->i", design, xtx_inv, design)
    if np.any(1.0 - leverage <= 0):
        raise ValueError("HC2 variance is undefined for this dataset")

    scaled = residuals**2 / (1.0 - leverage)
    meat = design.T @ (design * scaled[:, None])
    covariance = xtx_inv @ meat @ xtx_inv
    estimate = float(coefficients[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    df = len(sample) - np.linalg.matrix_rank(design)
    statistic = estimate / std_error
    p_value = float(2.0 * student_t.sf(abs(statistic), df))
    return {"estimate": estimate, "std_error": std_error, "p_value": p_value}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="CSV dataset path")
    parser.add_argument("--output", required=True, help="Path for the JSON result")
    args = parser.parse_args()
    try:
        result = estimate(args.data)
        with open(args.output, "w", encoding="utf-8") as output_file:
            json.dump(result, output_file, indent=2)
            output_file.write("\n")
    except Exception as exc:
        print(f"score.py: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
