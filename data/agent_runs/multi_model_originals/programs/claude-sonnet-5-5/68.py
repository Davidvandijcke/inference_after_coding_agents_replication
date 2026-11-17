#!/usr/bin/env python3
"""Rerun the single chosen analysis: effect of Pathways to Work on employment
12 months after random assignment (employed_m12), full sample.

Convention (CODEBOOK.md): OLS of the outcome on treatment + site indicators,
HC2 robust standard errors, two-sided t-test of the treatment coefficient.

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

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
    y = d[OUTCOME].to_numpy(float)
    sites = pd.get_dummies(d["site"], prefix="site", dtype=float)  # full set, no separate intercept
    X = np.column_stack([d["treatment"].to_numpy(float), sites.to_numpy(float)])
    n, k = X.shape

    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    h = np.einsum("ij,jk,ik->i", X, XtX_inv, X)  # leverage
    meat = (X * (resid**2 / (1 - h))[:, None]).T @ X  # HC2
    cov = XtX_inv @ meat @ XtX_inv

    est = float(beta[0])
    se = float(np.sqrt(cov[0, 0]))
    p = float(2 * stats.t.sf(abs(est / se), df=n - k))  # two-sided t-test

    with open(a.output, "w") as f:
        json.dump({"estimate": est, "std_error": se, "p_value": p}, f, indent=2)


if __name__ == "__main__":
    main()
