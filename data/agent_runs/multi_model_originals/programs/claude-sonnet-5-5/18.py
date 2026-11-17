#!/usr/bin/env python
"""Rerun the chosen Pathways to Work analysis on another dataset.

Chosen analysis (fixed a priori, no search): effect of treatment on employment
at 12 months (employed_m12), OLS on treatment + site indicators, HC2 robust
standard errors, two-sided t-test of the treatment coefficient.

Usage: python score.py --data <path> --output <path>
"""
import argparse, json
import numpy as np
import pandas as pd
from scipy import stats

OUTCOME = "employed_m12"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    d = pd.read_csv(a.data)
    d = d.dropna(subset=[OUTCOME, "treatment", "site"])
    y = d[OUTCOME].to_numpy(float)
    sites = pd.get_dummies(d["site"], prefix="site", dtype=float)  # full set = intercept
    X = np.column_stack([d["treatment"].to_numpy(float), sites.to_numpy()])
    n, k = X.shape

    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    h = np.einsum("ij,jk,ik->i", X, XtX_inv, X)          # leverages
    meat = X.T @ (X * (resid**2 / (1 - h))[:, None])      # HC2
    V = XtX_inv @ meat @ XtX_inv
    est, se = float(beta[0]), float(np.sqrt(V[0, 0]))
    p = float(2 * stats.t.sf(abs(est / se), df=n - k))

    with open(a.output, "w") as f:
        json.dump({"estimate": est, "std_error": se, "p_value": p}, f, indent=2)


if __name__ == "__main__":
    main()
