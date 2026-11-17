"""Pathways to Work: pre-specified primary analysis.

Effect of assignment to the program on earnings in the month before the
12-month follow-up (earnings_m12), full sample. OLS of earnings_m12 on
treatment plus site indicators; HC2 robust standard error; two-sided t-test
of the treatment coefficient (residual degrees of freedom).

Usage: python score.py --data <path.csv> --output <path.json>
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats

OUTCOME = "earnings_m12"


def estimate(df):
    df = df[[OUTCOME, "treatment", "site"]].dropna()
    y = df[OUTCOME].to_numpy(dtype=float)
    sites = np.sort(df["site"].unique())
    # treatment + one indicator per site (site indicators absorb the intercept)
    X = np.column_stack([df["treatment"].to_numpy(dtype=float)]
                        + [(df["site"].to_numpy() == s).astype(float) for s in sites])
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    leverage = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
    meat = (X * (resid ** 2 / (1.0 - leverage))[:, None]).T @ X
    vcov = XtX_inv @ meat @ XtX_inv
    est = float(beta[0])
    se = float(np.sqrt(vcov[0, 0]))
    p = float(2 * stats.t.sf(abs(est / se), n - k))
    return {"estimate": est, "std_error": se, "p_value": p}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = estimate(pd.read_csv(args.data))
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
