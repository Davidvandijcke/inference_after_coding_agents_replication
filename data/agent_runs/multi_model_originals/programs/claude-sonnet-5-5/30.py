#!/usr/bin/env python
"""Rerun the chosen analysis: ITT effect of Pathways to Work on employment at
12 months (employed_m12), OLS on treatment + site indicators, HC2 robust SE,
two-sided t-test on the treatment coefficient (CODEBOOK.md convention).

Usage: python score.py --data <path> --output <path>
"""
import argparse, json
import numpy as np, pandas as pd
from scipy import stats

OUTCOME = "employed_m12"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    d = pd.read_csv(a.data)
    y = d[OUTCOME].to_numpy(float)
    site = pd.get_dummies(d["site"]).to_numpy(float)  # full set of site dummies = intercept
    X = np.column_stack([d["treatment"].to_numpy(float), site])
    n, k = X.shape
    XtXi = np.linalg.inv(X.T @ X)
    beta = XtXi @ X.T @ y
    e = y - X @ beta
    h = np.einsum("ij,jk,ik->i", X, XtXi, X)
    a0 = X @ XtXi[:, 0]
    var = np.sum(a0**2 * e**2 / (1 - h))  # HC2 variance of treatment coefficient
    se = float(np.sqrt(var))
    t = beta[0] / se
    p = float(2 * stats.t.sf(abs(t), n - k))
    with open(a.output, "w") as f:
        json.dump({"estimate": float(beta[0]), "std_error": se, "p_value": p}, f, indent=2)

if __name__ == "__main__":
    main()
