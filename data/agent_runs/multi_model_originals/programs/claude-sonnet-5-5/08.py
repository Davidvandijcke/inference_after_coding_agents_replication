#!/usr/bin/env python
"""Rerun the single chosen analysis: effect of Pathways to Work on self-rated
health at 12 months (health_m12) among participants who do not live with a
child under 18 (parent == 0).

Convention (CODEBOOK.md): OLS of the outcome on treatment + site indicators,
HC2 robust standard errors, two-sided t-test of the treatment coefficient.

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats

OUTCOME = "health_m12"
SUBGROUP_VAR, SUBGROUP_VALUE = "parent", 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    d = pd.read_csv(a.data)
    d = d[d[SUBGROUP_VAR] == SUBGROUP_VALUE]

    sites = pd.get_dummies(d["site"], prefix="site", drop_first=True, dtype=float)
    X = np.column_stack([np.ones(len(d)), d["treatment"].to_numpy(float), sites.to_numpy()])
    y = d[OUTCOME].to_numpy(float)

    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    h = np.einsum("ij,jk,ik->i", X, XtX_inv, X)          # leverages
    meat = (X * (resid**2 / (1 - h))[:, None]).T @ X     # HC2
    cov = XtX_inv @ meat @ XtX_inv

    est, se = float(beta[1]), float(np.sqrt(cov[1, 1]))
    p = float(2 * stats.t.sf(abs(est / se), df=n - k))

    with open(a.output, "w") as f:
        json.dump({"estimate": est, "std_error": se, "p_value": p}, f, indent=2)


if __name__ == "__main__":
    main()
