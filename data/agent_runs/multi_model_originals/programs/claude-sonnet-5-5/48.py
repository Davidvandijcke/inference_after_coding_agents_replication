"""Rerun the single chosen analysis: effect of Pathways to Work on food insecurity
at 12 months (food_insecure_m12).

OLS of the outcome on treatment, site indicators and baseline covariates, with
HC2 heteroskedasticity-robust standard errors and a two-sided t-test of the
treatment coefficient (CODEBOOK.md convention). No searching is done here.

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats

OUTCOME = "food_insecure_m12"
COVARIATES = ["age", "female", "parent", "household_size", "employed_baseline",
              "earnings_baseline", "months_unemployed_baseline"]


def design(df):
    cols = [np.ones(len(df)), df["treatment"].to_numpy(float)]
    for s in sorted(df["site"].unique())[1:]:
        cols.append((df["site"] == s).to_numpy(float))
    for e in sorted(df["education"].unique())[1:]:
        cols.append((df["education"] == e).to_numpy(float))
    for c in COVARIATES:
        cols.append(df[c].to_numpy(float))
    return np.column_stack(cols)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    df = pd.read_csv(a.data)
    X = design(df)
    y = df[OUTCOME].to_numpy(float)
    n = len(y)
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    h = np.einsum("ij,jk,ik->i", X, XtX_inv, X)  # leverage
    meat = X.T @ (X * (resid**2 / (1 - h))[:, None])  # HC2
    cov = XtX_inv @ meat @ XtX_inv
    est, se = beta[1], np.sqrt(cov[1, 1])
    df_resid = n - np.linalg.matrix_rank(X)
    p = 2 * stats.t.sf(abs(est / se), df_resid)

    with open(a.output, "w") as f:
        json.dump({"estimate": float(est), "std_error": float(se), "p_value": float(p)}, f)


if __name__ == "__main__":
    main()
