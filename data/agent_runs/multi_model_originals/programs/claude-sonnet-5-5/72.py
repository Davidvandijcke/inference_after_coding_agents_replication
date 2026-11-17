#!/usr/bin/env python
"""Rerun the chosen Pathways to Work analysis on another dataset.

Chosen analysis: effect of treatment on training_m12 (enrolled in education or
job training at 12 months) among participants with parent == 1. OLS of the
outcome on treatment and site indicators, HC2 standard errors, two-sided
t-test of the treatment coefficient (CODEBOOK.md convention).

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats

OUTCOME = "training_m12"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    d = pd.read_csv(a.data)
    d = d[d["parent"] == 1]
    y = d[OUTCOME].to_numpy(float)
    sites = pd.get_dummies(d["site"]).to_numpy(float)  # full set of site dummies = intercept
    X = np.column_stack([d["treatment"].to_numpy(float), sites])

    XtXi = np.linalg.pinv(X.T @ X)
    beta = XtXi @ X.T @ y
    resid = y - X @ beta
    h = np.einsum("ij,jk,ik->i", X, XtXi, X)
    u = resid / np.sqrt(1 - h)  # HC2
    V = XtXi @ (X.T * u**2) @ X @ XtXi
    se = float(np.sqrt(V[0, 0]))
    df = len(y) - np.linalg.matrix_rank(X)
    t = beta[0] / se
    p = float(2 * stats.t.sf(abs(t), df))

    with open(a.output, "w") as f:
        json.dump({"estimate": float(beta[0]), "std_error": se, "p_value": p}, f, indent=2)


if __name__ == "__main__":
    main()
