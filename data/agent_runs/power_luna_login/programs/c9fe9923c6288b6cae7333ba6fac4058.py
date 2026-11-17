#!/usr/bin/env python3
"""Score the site-adjusted 12-month employment effect in a Pathways dataset."""

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="CSV with the evaluation columns")
    parser.add_argument("--output", required=True, help="Path for the JSON result")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = {"site", "treatment", "employed_m12"}
    missing = required.difference(data.columns)
    if missing:
        parser.error("missing required columns: " + ", ".join(sorted(missing)))
    if data[list(required)].isna().any().any():
        parser.error("site, treatment and employed_m12 must have no missing values")

    # OLS with an intercept, treatment, and site indicators (first observed site
    # omitted). Compute the HC2 sandwich variance directly.
    sites = pd.Categorical(data["site"])
    site_dummies = pd.get_dummies(sites, drop_first=True, dtype=float).to_numpy()
    x = np.column_stack(
        [np.ones(len(data)), data["treatment"].to_numpy(dtype=float), site_dummies]
    )
    y = data["employed_m12"].to_numpy(dtype=float)
    n, k = x.shape
    if n <= k or np.linalg.matrix_rank(x) != k:
        parser.error("design matrix must have full column rank and positive residual df")

    xtx_inv = np.linalg.inv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    residuals = y - x @ beta
    leverage = np.einsum("ij,jk,ik->i", x, xtx_inv, x)
    adjusted_sq = residuals**2 / (1.0 - leverage)
    meat = x.T @ (x * adjusted_sq[:, None])
    covariance = xtx_inv @ meat @ xtx_inv
    estimate = float(beta[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    statistic = estimate / std_error
    p_value = float(2.0 * t.sf(abs(statistic), df=n - k))

    with open(args.output, "w", encoding="utf-8") as output:
        json.dump(
            {"estimate": estimate, "std_error": std_error, "p_value": p_value},
            output,
            indent=2,
        )
        output.write("\n")


if __name__ == "__main__":
    main()
