#!/usr/bin/env python3
"""Rerun the single chosen analysis (no search).

Effect of Pathways to Work on 12-month self-rated health (health_m12) among
participants aged 45 or older: OLS of health_m12 on treatment + site indicators,
HC2 standard errors, two-sided t-test of the treatment coefficient.

Usage: python score.py --data <path> --output <path>
"""
import argparse, json
import numpy as np
import pandas as pd
from scipy import stats

AGE_CUTOFF = 45
OUTCOME = "health_m12"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    d = pd.read_csv(a.data)
    d = d[d["age"] >= AGE_CUTOFF]
    sites = sorted(d["site"].unique())
    X = np.column_stack([d["treatment"].astype(float)] +
                        [(d["site"] == s).astype(float) for s in sites])
    y = d[OUTCOME].astype(float).to_numpy()

    XtXi = np.linalg.pinv(X.T @ X)
    beta = XtXi @ X.T @ y
    resid = y - X @ beta
    h = np.einsum("ij,jk,ik->i", X, XtXi, X)
    meat = (X * (resid**2 / (1 - h))[:, None]).T @ X
    V = XtXi @ meat @ XtXi
    se = float(np.sqrt(V[0, 0]))
    dof = len(y) - np.linalg.matrix_rank(X)
    p = float(2 * stats.t.sf(abs(beta[0] / se), dof))

    with open(a.output, "w") as f:
        json.dump({"estimate": float(beta[0]), "std_error": se, "p_value": p}, f, indent=2)


if __name__ == "__main__":
    main()
