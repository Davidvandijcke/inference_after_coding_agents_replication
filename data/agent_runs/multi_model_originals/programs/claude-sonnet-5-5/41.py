#!/usr/bin/env python
"""Re-run the single chosen Pathways to Work analysis on another dataset.

Chosen test (exploratory in the original data): effect of `treatment` on
`hours_m6` among participants with education == 3 (some college), by OLS with
site indicators, HC2 robust standard errors, two-sided t-test of the treatment
coefficient (CODEBOOK.md convention). No search over outcomes or subgroups.

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats

OUTCOME = "hours_m6"
SUBGROUP_COL, SUBGROUP_VALUE = "education", 3


def run(df):
    s = df[df[SUBGROUP_COL] == SUBGROUP_VALUE]
    y = s[OUTCOME].to_numpy(float)
    sites = pd.get_dummies(s["site"], dtype=float).to_numpy()
    X = np.column_stack([s["treatment"].to_numpy(float), sites])
    n, k = X.shape
    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    resid = y - X @ beta
    h = np.einsum("ij,jk,ik->i", X, xtx_inv, X)
    u = resid / np.sqrt(1.0 - h)  # HC2
    cov = xtx_inv @ (X.T * u**2) @ X @ xtx_inv
    est = float(beta[0])
    se = float(np.sqrt(cov[0, 0]))
    p = float(2 * stats.t.sf(abs(est / se), n - np.linalg.matrix_rank(X)))
    return {"estimate": est, "std_error": se, "p_value": p}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    res = run(pd.read_csv(a.data))
    with open(a.output, "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
