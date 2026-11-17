#!/usr/bin/env python
"""Pathways to Work: pre-specified re-test of the single exploratory finding.

Outcome:  food_insecure_m12 (household food insecurity at 12 months)
Subgroup: participants aged 45 or older at random assignment
Model:    OLS of outcome on treatment + site indicators, HC2 robust SEs,
          two-sided t-test of the treatment coefficient (CODEBOOK convention).
No search is performed; only this one test is run.

Usage: python score.py --data <path> --output <path>
"""
import argparse, json
import numpy as np
import pandas as pd
from scipy import stats

OUTCOME, AGE_MIN = "food_insecure_m12", 45


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    d = pd.read_csv(a.data)
    g = d[d["age"] >= AGE_MIN]
    y = g[OUTCOME].to_numpy(float)
    site = pd.get_dummies(g["site"].astype(int), prefix="s").astype(float)
    X = np.column_stack([g["treatment"].to_numpy(float), site.to_numpy()])  # site dummies span the intercept
    n, k = X.shape

    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    h = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
    meat = (X * (resid ** 2 / (1 - h))[:, None]).T @ X          # HC2
    se = float(np.sqrt((XtX_inv @ meat @ XtX_inv)[0, 0]))
    est = float(beta[0])
    p = float(2 * stats.t.sf(abs(est / se), n - k))

    with open(a.output, "w") as f:
        json.dump({"estimate": est, "std_error": se, "p_value": p}, f, indent=2)


if __name__ == "__main__":
    main()
