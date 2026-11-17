"""Pathways to Work: effect of program assignment on 12-month earnings.

Reruns the single pre-chosen analysis (no search):
  OLS of earnings_m12 on treatment + site indicators + baseline covariates
  (age, female, parent, education indicators, household_size,
  employed_baseline, earnings_baseline, months_unemployed_baseline),
  HC2 robust standard errors, two-sided t-test of the treatment coefficient.

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats

OUTCOME = "earnings_m12"
CONTINUOUS = ["age", "female", "parent", "household_size", "employed_baseline",
              "earnings_baseline", "months_unemployed_baseline"]
CATEGORICAL = ["site", "education"]


def design(df):
    cols = [np.ones(len(df)), df["treatment"].to_numpy(float)]
    for c in CONTINUOUS:
        cols.append(df[c].to_numpy(float))
    for c in CATEGORICAL:
        for level in sorted(df[c].unique())[1:]:
            cols.append((df[c] == level).to_numpy(float))
    return np.column_stack(cols)


def ols_hc2(X, y):
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    h = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
    meat = (X * (resid ** 2 / (1 - h))[:, None]).T @ X
    vcov = XtX_inv @ meat @ XtX_inv
    df_resid = len(y) - np.linalg.matrix_rank(X)
    return beta, vcov, df_resid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    df = df.dropna(subset=[OUTCOME, "treatment"] + CONTINUOUS + CATEGORICAL)
    X = design(df)
    y = df[OUTCOME].to_numpy(float)
    beta, vcov, df_resid = ols_hc2(X, y)

    estimate = float(beta[1])
    std_error = float(np.sqrt(vcov[1, 1]))
    t = estimate / std_error
    p_value = float(2 * stats.t.sf(abs(t), df_resid))

    result = {"estimate": estimate, "std_error": std_error, "p_value": p_value,
              "outcome": OUTCOME, "n": int(len(df)), "df_resid": int(df_resid)}
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
