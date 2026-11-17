#!/usr/bin/env python3
"""Rerun the chosen Pathways to Work analysis on another dataset.

Chosen (exploratory) analysis: effect of assignment to the program on
employment 12 months after random assignment (employed_m12) among participants
at site 3. Per the evaluation convention: OLS of the outcome on the treatment
indicator (site indicators are constant within a single site, so the model is
treatment + intercept), HC2 robust standard errors, two-sided t-test of the
treatment coefficient.

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats

SITE = 3
OUTCOME = "employed_m12"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    d = pd.read_csv(args.data)
    d = d[d["site"] == SITE]
    y = d[OUTCOME].to_numpy(float)
    X = np.column_stack([np.ones(len(d)), d["treatment"].to_numpy(float)])
    n, k = X.shape

    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    resid = y - X @ beta
    h = np.einsum("ij,jk,ik->i", X, xtx_inv, X)
    meat = (X * (resid / np.sqrt(1 - h))[:, None]).T @ (X * (resid / np.sqrt(1 - h))[:, None])
    cov = xtx_inv @ meat @ xtx_inv

    est = float(beta[1])
    se = float(np.sqrt(cov[1, 1]))
    p = float(2 * stats.t.sf(abs(est / se), df=n - k))
    with open(args.output, "w") as f:
        json.dump({"estimate": est, "std_error": se, "p_value": p}, f, indent=2)


if __name__ == "__main__":
    main()
