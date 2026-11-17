#!/usr/bin/env python3
"""Rerun the single chosen analysis (no search).

Effect of Pathways to Work on `debt_collections_m12` among participants employed
at baseline (employed_baseline == 1): OLS of the outcome on treatment and site
indicators, HC2 standard errors, two-sided t-test on the treatment coefficient
(df = n - k).

Usage: python score.py --data <path> --output <path>
"""
import argparse, json
import numpy as np
import pandas as pd
from scipy import stats

OUTCOME = "debt_collections_m12"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    d = pd.read_csv(a.data)
    d = d[d["employed_baseline"] == 1]
    y = d[OUTCOME].to_numpy(float)
    sites = sorted(d["site"].unique())
    cols = [d["treatment"].to_numpy(float)] + [(d["site"] == s).to_numpy(float) for s in sites]  # site dummies = intercept
    X = np.column_stack(cols)
    n, k = X.shape
    XtXi = np.linalg.inv(X.T @ X)
    beta = XtXi @ X.T @ y
    e = y - X @ beta
    h = np.einsum("ij,jk,ik->i", X, XtXi, X)
    meat = (X * (e**2 / (1 - h))[:, None]).T @ X
    V = XtXi @ meat @ XtXi
    est, se = beta[0], float(np.sqrt(V[0, 0]))
    p = float(2 * stats.t.sf(abs(est / se), n - k))
    with open(a.output, "w") as f:
        json.dump({"estimate": float(est), "std_error": se, "p_value": p}, f, indent=2)


if __name__ == "__main__":
    main()
