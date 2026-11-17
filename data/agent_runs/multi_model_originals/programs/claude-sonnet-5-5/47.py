#!/usr/bin/env python
"""Re-run the single chosen analysis: effect of Pathways to Work on self-rated
health at 6 months (health_m6), full sample.

Convention (CODEBOOK.md): OLS of the outcome on `treatment` plus site
indicators, HC2 robust standard errors, two-sided t-test on `treatment`.
No search is performed; usage:
    python score.py --data <path> --output <path>
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats

OUTCOME = "health_m6"


def estimate(df):
    y = df[OUTCOME].to_numpy(float)
    sites = np.sort(df["site"].unique())
    X = np.column_stack(
        [df["treatment"].to_numpy(float), np.ones(len(df))]
        + [(df["site"].to_numpy() == s).astype(float) for s in sites[1:]]
    )
    n, k = X.shape
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    resid = y - X @ beta
    h = np.einsum("ij,jk,ik->i", X, xtx_inv, X)
    meat = (X * (resid / (1 - h) ** 0.5)[:, None]).T @ (X * (resid / (1 - h) ** 0.5)[:, None])
    cov = xtx_inv @ meat @ xtx_inv
    se = float(np.sqrt(cov[0, 0]))
    t = beta[0] / se
    p = float(2 * stats.t.sf(abs(t), n - k))
    return float(beta[0]), se, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    df = pd.read_csv(a.data)
    b, se, p = estimate(df)
    with open(a.output, "w") as f:
        json.dump({"estimate": b, "std_error": se, "p_value": p}, f, indent=2)


if __name__ == "__main__":
    main()
