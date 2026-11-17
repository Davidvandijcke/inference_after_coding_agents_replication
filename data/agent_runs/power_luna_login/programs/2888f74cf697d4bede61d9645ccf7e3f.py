#!/usr/bin/env python3
"""Score the prespecified exploratory subgroup result for Pathways to Work.

Estimates the treatment effect on 12-month employment among participants
employed at baseline, with site fixed effects and HC2 standard errors.
"""

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


def score(data_path):
    data = pd.read_csv(data_path)
    subgroup = data.loc[data["employed_baseline"] == 1].copy()
    y = subgroup["employed_m12"].to_numpy(dtype=float)
    treatment = subgroup["treatment"].to_numpy(dtype=float)
    site = pd.get_dummies(subgroup["site"], drop_first=True, dtype=float)
    X = np.column_stack(
        [np.ones(len(subgroup)), treatment, site.to_numpy(dtype=float)]
    )

    if len(y) <= X.shape[1] or np.unique(treatment).size != 2:
        raise ValueError("Insufficient observations or treatment variation in subgroup")

    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    residuals = y - X @ beta
    leverage = np.einsum("ij,jk,ik->i", X, xtx_inv, X)
    df_resid = len(y) - np.linalg.matrix_rank(X)

    # HC2 sandwich covariance: X' diag(e_i^2 / (1-h_i)) X.
    adjusted_sq_resid = residuals**2 / (1.0 - leverage)
    meat = X.T @ (X * adjusted_sq_resid[:, None])
    covariance = xtx_inv @ meat @ xtx_inv
    estimate = float(beta[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    statistic = estimate / std_error
    p_value = float(2.0 * student_t.sf(abs(statistic), df_resid))

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
