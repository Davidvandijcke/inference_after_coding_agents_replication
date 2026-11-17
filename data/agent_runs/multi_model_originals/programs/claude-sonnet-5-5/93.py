#!/usr/bin/env python
"""Pathways to Work: effect of assignment to the program on employment at 12 months.

OLS of employed_m12 on treatment + site indicators, HC2 standard errors,
two-sided t-test (n - k degrees of freedom) of the treatment coefficient.
Runs only this pre-chosen analysis; no search over outcomes or subgroups.
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
    X = np.column_stack([d["treatment"].to_numpy(float),
                         pd.get_dummies(d["site"]).to_numpy(float)])  # site dummies serve as intercept
    n, k = X.shape
    XtXi = np.linalg.inv(X.T @ X)
    beta = XtXi @ X.T @ y
    e = y - X @ beta
    h = np.einsum("ij,jk,ik->i", X, XtXi, X)
    meat = (X * (e**2 / (1 - h))[:, None]).T @ X
    V = XtXi @ meat @ XtXi
    se = float(np.sqrt(V[0, 0]))
    t = beta[0] / se
    p = float(2 * stats.t.sf(abs(t), n - k))
    with open(a.output, "w") as f:
        json.dump({"estimate": float(beta[0]), "std_error": se, "p_value": p}, f, indent=2)

if __name__ == "__main__":
    main()
