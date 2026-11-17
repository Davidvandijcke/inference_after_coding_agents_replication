#!/usr/bin/env python3
"""Pathways to Work: pre-chosen analysis (no search).

Effect of assignment to the program on psychological distress (Kessler-6) at
6 months, full sample: OLS of distress_m6 on treatment + site indicators,
HC2 robust standard errors, two-sided t-test on the treatment coefficient.

Usage: python score.py --data <path> --output <path>
"""
import argparse, json
import numpy as np
import pandas as pd
from scipy import stats

OUTCOME = "distress_m6"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    d = pd.read_csv(a.data)
    y = d[OUTCOME].to_numpy(float)
    site = pd.get_dummies(d["site"], prefix="site").astype(float)  # full set = intercept
    X = np.column_stack([d["treatment"].to_numpy(float), site.to_numpy()])
    n, k = X.shape

    XtXi = np.linalg.inv(X.T @ X)
    beta = XtXi @ X.T @ y
    e = y - X @ beta
    h = np.einsum("ij,jk,ik->i", X, XtXi, X)
    meat = X.T @ (X * (e**2 / (1 - h))[:, None])  # HC2
    V = XtXi @ meat @ XtXi
    est = float(beta[0])
    se = float(np.sqrt(V[0, 0]))
    p = float(2 * stats.t.sf(abs(est / se), n - k))

    with open(a.output, "w") as f:
        json.dump({"estimate": est, "std_error": se, "p_value": p}, f, indent=2)


if __name__ == "__main__":
    main()
