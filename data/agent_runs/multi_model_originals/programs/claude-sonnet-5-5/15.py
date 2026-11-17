#!/usr/bin/env python
"""Re-run the single chosen analysis: effect of Pathways to Work on earnings at
12 months (earnings_m12) among participants aged 40 or older.

Convention (CODEBOOK.md): OLS of the outcome on treatment + site indicators within
the subgroup, HC2 robust standard errors, two-sided t-test on the treatment
coefficient. No search is performed; the subgroup and outcome are fixed.

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats

OUTCOME = "earnings_m12"
AGE_CUTOFF = 40


def hc2_treatment_effect(df, outcome):
    y = df[outcome].to_numpy(float)
    t = df["treatment"].to_numpy(float)
    sites = pd.get_dummies(df["site"], dtype=float)
    X = np.column_stack([t, sites.to_numpy()])  # site dummies span the intercept
    X = X[:, np.abs(X).sum(axis=0) > 0]
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    h = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
    meat = (X * (resid**2 / (1 - h))[:, None]).T @ X
    V = XtX_inv @ meat @ XtX_inv
    est = float(beta[0])
    se = float(np.sqrt(V[0, 0]))
    dof = n - np.linalg.matrix_rank(X)
    p = float(2 * stats.t.sf(abs(est / se), dof)) if se > 0 else float("nan")
    return est, se, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    df = pd.read_csv(a.data)
    sub = df[df["age"] >= AGE_CUTOFF]
    est, se, p = hc2_treatment_effect(sub, OUTCOME)
    with open(a.output, "w") as f:
        json.dump({"estimate": est, "std_error": se, "p_value": p}, f)


if __name__ == "__main__":
    main()
