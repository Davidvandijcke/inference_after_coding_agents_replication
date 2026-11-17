#!/usr/bin/env python
"""Rerun the single chosen analysis: effect of Pathways to Work on 6-month
psychological distress (distress_m6) among participants at site 1.

Convention (CODEBOOK.md): OLS of the outcome on the treatment indicator and
site indicators (constant within a single site), HC2 robust standard errors,
two-sided t-test of the treatment coefficient.

Usage: python score.py --data <path> --output <path>
Writes JSON with estimate, std_error, p_value.
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats

SITE = 1
OUTCOME = "distress_m6"


def hc2_ols(y, X, j):
    """Return coefficient, HC2 standard error and two-sided t-test p-value for column j."""
    n, k = X.shape
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    resid = y - X @ beta
    h = np.einsum("ij,jk,ik->i", X, xtx_inv, X)
    meat = X.T @ (X * (resid**2 / (1.0 - h))[:, None])
    cov = xtx_inv @ meat @ xtx_inv
    se = float(np.sqrt(cov[j, j]))
    p = float(2 * stats.t.sf(abs(beta[j] / se), n - k))
    return float(beta[j]), se, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    sub = df[df["site"] == SITE]
    y = sub[OUTCOME].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(sub)), sub["treatment"].to_numpy(dtype=float)])
    est, se, p = hc2_ols(y, X, 1)

    with open(args.output, "w") as f:
        json.dump({"estimate": est, "std_error": se, "p_value": p}, f, indent=2)


if __name__ == "__main__":
    main()
