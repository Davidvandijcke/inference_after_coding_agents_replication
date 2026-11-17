#!/usr/bin/env python
"""Rerun the chosen Pathways to Work analysis on a dataset with the codebook columns.

Chosen test (selected during exploratory analysis of the original data):
  Outcome   : food insecurity averaged over the 6- and 12-month follow-ups,
              (food_insecure_m6 + food_insecure_m12) / 2
  Sample    : all participants
  Model     : OLS of the outcome on treatment, site indicators, and baseline
              covariates (age, female, parent, education indicators,
              household_size, employed_baseline, earnings_baseline,
              months_unemployed_baseline)
  Inference : HC2 robust standard error, two-sided t-test of the treatment
              coefficient (residual degrees of freedom)

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats

COVARIATES = ["age", "female", "parent", "household_size", "employed_baseline",
              "earnings_baseline", "months_unemployed_baseline"]


def design(df):
    cols = [np.ones(len(df)), df["treatment"].to_numpy(float)]
    for s in sorted(df["site"].unique())[1:]:
        cols.append((df["site"] == s).to_numpy(float))
    cols += [df[c].to_numpy(float) for c in COVARIATES]
    for e in sorted(df["education"].unique())[1:]:
        cols.append((df["education"] == e).to_numpy(float))
    return np.column_stack(cols)


def ols_hc2(y, X):
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    leverage = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
    omega = resid ** 2 / (1.0 - leverage)
    vcov = XtX_inv @ (X.T * omega) @ X @ XtX_inv
    df_resid = len(y) - np.linalg.matrix_rank(X)
    return beta, vcov, df_resid


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    needed = ["treatment", "site", "education", "food_insecure_m6", "food_insecure_m12"] + COVARIATES
    df = df.dropna(subset=needed).reset_index(drop=True)

    y = ((df["food_insecure_m6"] + df["food_insecure_m12"]) / 2).to_numpy(float)
    X = design(df)
    beta, vcov, df_resid = ols_hc2(y, X)

    estimate = float(beta[1])
    std_error = float(np.sqrt(vcov[1, 1]))
    p_value = float(2 * stats.t.sf(abs(estimate / std_error), df_resid))

    with open(args.output, "w") as f:
        json.dump({"estimate": estimate, "std_error": std_error, "p_value": p_value}, f, indent=2)


if __name__ == "__main__":
    main()
