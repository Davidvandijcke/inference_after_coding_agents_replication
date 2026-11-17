#!/usr/bin/env python
"""Rerun the chosen Pathways to Work analysis on a dataset with the same columns.

Chosen analysis (fixed in advance, no search):
    OLS of distress_m12 (Kessler-6 psychological distress at 12 months)
    on treatment + site indicators, full sample, no baseline covariates.
    HC2 heteroskedasticity-robust standard error; two-sided t-test of a
    zero coefficient on treatment with n - k residual degrees of freedom.

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats

OUTCOME = "distress_m12"


def hc2_treatment_effect(df):
    df = df[[OUTCOME, "treatment", "site"]].dropna()
    y = df[OUTCOME].to_numpy(dtype=float)
    sites = pd.get_dummies(df["site"].astype("category"), prefix="site", drop_first=True, dtype=float)
    X = np.column_stack([np.ones(len(df)), df["treatment"].to_numpy(dtype=float), sites.to_numpy()])
    n, k = X.shape

    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    leverage = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
    omega = resid ** 2 / (1.0 - leverage)
    vcov = XtX_inv @ (X.T * omega) @ X @ XtX_inv

    est = float(beta[1])
    se = float(np.sqrt(vcov[1, 1]))
    p = float(2 * stats.t.sf(abs(est / se), df=n - k))
    return {"estimate": est, "std_error": se, "p_value": p}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="CSV with the Pathways to Work columns")
    ap.add_argument("--output", required=True, help="Path for the JSON result")
    args = ap.parse_args()

    result = hc2_treatment_effect(pd.read_csv(args.data))
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
