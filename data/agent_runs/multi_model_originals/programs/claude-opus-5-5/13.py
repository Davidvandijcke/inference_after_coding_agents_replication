"""Rerun the chosen Pathways to Work analysis on a dataset with the codebook columns.

Chosen test (fixed in advance; no search is performed here):
    Effect of random assignment to the program on earnings in the month before
    the 12-month survey (earnings_m12), full sample.
    OLS of earnings_m12 on treatment + site indicators + baseline covariates
    (age, female, parent, education indicators, household_size,
    employed_baseline, earnings_baseline, months_unemployed_baseline).
    HC2 robust standard error; two-sided t-test of the treatment coefficient
    with n - k residual degrees of freedom.

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats

OUTCOME = "earnings_m12"
NUMERIC_COVARIATES = [
    "age", "female", "parent", "household_size", "employed_baseline",
    "earnings_baseline", "months_unemployed_baseline",
]
CATEGORICAL_COVARIATES = ["site", "education"]


def design_matrix(df):
    cols = [np.ones(len(df)), df["treatment"].to_numpy(float)]
    for v in CATEGORICAL_COVARIATES:
        for level in sorted(df[v].unique())[1:]:
            cols.append((df[v] == level).to_numpy(float))
    for v in NUMERIC_COVARIATES:
        cols.append(df[v].to_numpy(float))
    return np.column_stack(cols)


def ols_hc2(y, X):
    n, k = X.shape
    if np.linalg.matrix_rank(X) < k:
        raise ValueError("design matrix is rank deficient")
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    resid = y - X @ beta
    leverage = np.einsum("ij,jk,ik->i", X, xtx_inv, X)
    meat = (X * (resid**2 / (1.0 - leverage))[:, None]).T @ X
    vcov = xtx_inv @ meat @ xtx_inv
    return beta, np.sqrt(np.diag(vcov)), n - k


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    needed = [OUTCOME, "treatment"] + CATEGORICAL_COVARIATES + NUMERIC_COVARIATES
    df = df[needed].dropna()

    beta, se, df_resid = ols_hc2(df[OUTCOME].to_numpy(float), design_matrix(df))
    estimate, std_error = float(beta[1]), float(se[1])
    p_value = float(2 * stats.t.sf(abs(estimate / std_error), df_resid))

    with open(args.output, "w") as f:
        json.dump({"estimate": estimate, "std_error": std_error, "p_value": p_value}, f, indent=2)


if __name__ == "__main__":
    main()
