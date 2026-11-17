"""Pathways to Work: pre-specified effect of program assignment on 12-month earnings.

Model: OLS of earnings_m12 on treatment, site indicators, and a fixed set of
baseline covariates; HC2 robust standard error; two-sided t-test (df = n - k)
of a zero coefficient on treatment.

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats

OUTCOME = "earnings_m12"
COVARIATES = [
    "age", "female", "parent", "education", "household_size",
    "employed_baseline", "earnings_baseline", "months_unemployed_baseline",
]


def design(df):
    sites = pd.get_dummies(df["site"].astype("category"), prefix="site", drop_first=True, dtype=float)
    X = pd.concat(
        [pd.Series(1.0, index=df.index, name="const"), df[["treatment"]].astype(float),
         sites, df[COVARIATES].astype(float)],
        axis=1,
    )
    return X


def ols_hc2(y, X):
    Xv, yv = X.to_numpy(), y.to_numpy(dtype=float)
    n, k = Xv.shape
    XtX_inv = np.linalg.pinv(Xv.T @ Xv)
    beta = XtX_inv @ Xv.T @ yv
    resid = yv - Xv @ beta
    lev = np.einsum("ij,jk,ik->i", Xv, XtX_inv, Xv)
    w = resid ** 2 / (1.0 - lev)
    V = XtX_inv @ (Xv.T * w) @ Xv @ XtX_inv
    return beta, V, n - k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    df = df[[OUTCOME, "treatment", "site"] + COVARIATES].dropna()
    X = design(df)
    beta, V, dof = ols_hc2(df[OUTCOME], X)
    j = X.columns.get_loc("treatment")
    est, se = float(beta[j]), float(np.sqrt(V[j, j]))
    p = float(2 * stats.t.sf(abs(est / se), dof))

    with open(args.output, "w") as f:
        json.dump({"estimate": est, "std_error": se, "p_value": p}, f, indent=2)


if __name__ == "__main__":
    main()
