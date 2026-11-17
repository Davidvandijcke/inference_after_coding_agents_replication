#!/usr/bin/env python
"""Pathways to Work: rerun the chosen analysis on a dataset with the codebook columns.

Chosen test (fixed in advance of any new data; no specification search here):
  outcome   employed_m12 (employed in the survey week, 12 months after assignment)
  sample    all participants
  model     OLS of employed_m12 on treatment + site indicators
  inference HC2 heteroskedasticity-robust SE, two-sided t-test of the treatment
            coefficient with n - k residual degrees of freedom

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats

OUTCOME = "employed_m12"


def estimate(df):
    df = df[[OUTCOME, "treatment", "site"]].dropna()
    y = df[OUTCOME].to_numpy(dtype=float)
    sites = pd.get_dummies(df["site"].astype("category"), drop_first=True, dtype=float)
    X = np.column_stack([np.ones(len(df)), df["treatment"].to_numpy(dtype=float), sites.to_numpy()])
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    leverage = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
    meat = (X * (resid**2 / (1.0 - leverage))[:, None]).T @ X
    vcov = XtX_inv @ meat @ XtX_inv
    est = float(beta[1])
    se = float(np.sqrt(vcov[1, 1]))
    p = float(2 * stats.t.sf(abs(est / se), df=n - k))
    return {"estimate": est, "std_error": se, "p_value": p}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    result = estimate(pd.read_csv(args.data))
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
