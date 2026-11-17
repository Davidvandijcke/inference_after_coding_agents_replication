"""Pathways to Work: rerun the single chosen analysis on another dataset.

Chosen test: effect of `treatment` on `employed_m12` among participants with
months_unemployed_baseline >= 6, estimated by OLS with site indicators, HC2
standard errors, and a two-sided t-test of the treatment coefficient
(CODEBOOK.md convention). No search is performed.

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats

OUTCOME = "employed_m12"
MIN_MONTHS_UNEMPLOYED = 6


def estimate(df):
    sub = df[df["months_unemployed_baseline"] >= MIN_MONTHS_UNEMPLOYED]
    y = sub[OUTCOME].to_numpy(float)
    t = sub["treatment"].to_numpy(float)
    sites = pd.get_dummies(sub["site"], dtype=float).to_numpy()  # full set acts as intercept
    X = np.column_stack([t, sites])
    n, k = X.shape
    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    resid = y - X @ beta
    h = np.einsum("ij,jk,ik->i", X, xtx_inv, X)
    # HC2: squared residual / (1 - h)
    xu = X * (resid / np.sqrt(1 - h))[:, None]
    meat = xu.T @ xu
    cov = xtx_inv @ meat @ xtx_inv
    se = float(np.sqrt(cov[0, 0]))
    df_resid = n - np.linalg.matrix_rank(X)
    p = float(2 * stats.t.sf(abs(beta[0] / se), df_resid))
    return float(beta[0]), se, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    est, se, p = estimate(pd.read_csv(a.data))
    with open(a.output, "w") as f:
        json.dump({"estimate": est, "std_error": se, "p_value": p}, f, indent=2)


if __name__ == "__main__":
    main()
