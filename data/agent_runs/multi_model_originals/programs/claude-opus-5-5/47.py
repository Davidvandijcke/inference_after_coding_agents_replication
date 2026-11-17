"""Rerun the chosen Pathways to Work analysis on a dataset with the same columns.

Chosen test: effect of assignment to the program on self-rated health
(health_m6, 1-5) at the 6-month follow-up, full sample. OLS of health_m6 on
treatment and site indicators, HC2 robust standard errors, two-sided t-test of
the treatment coefficient (CODEBOOK.md analysis convention).

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats


def estimate(df):
    df = df[["health_m6", "treatment", "site"]].dropna()
    y = df["health_m6"].to_numpy(float)
    sites = pd.get_dummies(df["site"].astype("category"), drop_first=True, dtype=float)
    X = np.column_stack([np.ones(len(df)), df["treatment"].to_numpy(float), sites.to_numpy()])
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    leverage = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
    meat = X.T @ (X * (resid**2 / (1 - leverage))[:, None])
    vcov = XtX_inv @ meat @ XtX_inv
    est = float(beta[1])
    se = float(np.sqrt(vcov[1, 1]))
    p = float(2 * stats.t.sf(abs(est / se), df=n - k))
    return {"estimate": est, "std_error": se, "p_value": p}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = estimate(pd.read_csv(args.data))
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
