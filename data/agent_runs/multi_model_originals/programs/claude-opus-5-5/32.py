#!/usr/bin/env python
"""Pathways to Work: chosen confirmatory analysis.

Effect of random assignment on 12-month monthly earnings (earnings_m12), full
sample. OLS of earnings_m12 on treatment, site indicators, and the eight
baseline covariates in CODEBOOK.md; HC2 robust standard error; two-sided
t-test of the treatment coefficient (residual df = n - k).

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


def estimate(df):
    cols = [OUTCOME, "treatment", "site"] + COVARIATES
    df = df[cols].dropna()
    y = df[OUTCOME].to_numpy(float)
    sites = pd.get_dummies(df["site"].astype("category"), prefix="site", drop_first=True, dtype=float)
    X = pd.concat(
        [pd.Series(1.0, index=df.index, name="const"), df["treatment"].astype(float), sites, df[COVARIATES].astype(float)],
        axis=1,
    ).to_numpy(float)
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    leverage = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
    omega = resid ** 2 / (1.0 - leverage)  # HC2
    vcov = XtX_inv @ (X.T * omega) @ X @ XtX_inv
    b, se = beta[1], float(np.sqrt(vcov[1, 1]))
    p = 2 * stats.t.sf(abs(b / se), df=n - k)
    return {"estimate": float(b), "std_error": se, "p_value": float(p)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    res = estimate(pd.read_csv(a.data))
    with open(a.output, "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
