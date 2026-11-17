#!/usr/bin/env python
"""Rerun the single chosen analysis: effect of Pathways to Work on 12-month
psychological distress (distress_m12), OLS on treatment + site indicators,
HC2 standard errors, two-sided t-test on the treatment coefficient.

Usage: python score.py --data <path> --output <path>
"""
import argparse, json
import numpy as np, pandas as pd
from scipy import stats

OUTCOME = "distress_m12"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    d = pd.read_csv(a.data)
    sites = pd.get_dummies(d["site"], prefix="site", drop_first=True).astype(float)
    X = np.column_stack([np.ones(len(d)), d["treatment"].astype(float).values, sites.values])
    y = d[OUTCOME].astype(float).values
    xtxi = np.linalg.inv(X.T @ X)
    beta = xtxi @ X.T @ y
    resid = y - X @ beta
    h = np.einsum("ij,jk,ik->i", X, xtxi, X)
    meat = (X * (resid**2 / (1 - h))[:, None]).T @ X  # HC2
    V = xtxi @ meat @ xtxi
    est, se = float(beta[1]), float(np.sqrt(V[1, 1]))
    p = float(2 * stats.t.sf(abs(est / se), len(y) - X.shape[1]))
    with open(a.output, "w") as f:
        json.dump({"estimate": est, "std_error": se, "p_value": p}, f, indent=2)


if __name__ == "__main__":
    main()
