#!/usr/bin/env python3
"""Score the block-adjusted treatment effect on immediate pension ranking."""

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import t


OUTCOME = "rank_pension_post"


def score(data_path):
    data = pd.read_csv(data_path)
    sample = data[["treatment", "block", OUTCOME]].dropna().copy()
    if sample.empty:
        raise ValueError(f"No complete observations for {OUTCOME}")

    # OLS with an intercept, treatment indicator, and block fixed effects.
    block_effects = pd.get_dummies(
        sample["block"].astype("category"), drop_first=True, dtype=float
    )
    design = np.column_stack(
        [np.ones(len(sample)), sample["treatment"].to_numpy(dtype=float), block_effects]
    )
    outcome = sample[OUTCOME].to_numpy(dtype=float)
    n, k = design.shape
    if n <= k or np.linalg.matrix_rank(design) < k:
        raise ValueError("Insufficient complete observations or non-identifiable design")

    xtx_inv = np.linalg.inv(design.T @ design)
    coefficients = xtx_inv @ design.T @ outcome
    residuals = outcome - design @ coefficients

    # HC2 sandwich covariance: (X'X)^-1 X' diag(e_i^2/(1-h_ii)) X (X'X)^-1.
    leverage = np.einsum("ij,jk,ik->i", design, xtx_inv, design)
    adjusted_sq_residuals = residuals**2 / (1.0 - leverage)
    meat = design.T @ (design * adjusted_sq_residuals[:, None])
    covariance = xtx_inv @ meat @ xtx_inv

    treatment_index = 1
    estimate = float(coefficients[treatment_index])
    std_error = float(np.sqrt(covariance[treatment_index, treatment_index]))
    degrees_freedom = n - k
    statistic = estimate / std_error
    p_value = float(2.0 * t.sf(abs(statistic), degrees_freedom))
    return {
        "estimate": estimate,
        "std_error": std_error,
        "p_value": p_value,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    result = score(args.data)
    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2)
        output_file.write("\n")


if __name__ == "__main__":
    main()
