#!/usr/bin/env python
"""Re-run the single chosen (exploratory) analysis on another dataset.

Chosen test: effect of `treatment` on `debt_collections_m6` among participants
with education >= 3 (some college or more), by OLS with site indicators,
HC2 robust standard errors, two-sided t-test on the treatment coefficient
(CODEBOOK.md convention). No search is performed.

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats

OUTCOME = "debt_collections_m6"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    d = pd.read_csv(args.data)
    d = d[d["education"] >= 3]
    y = d[OUTCOME].to_numpy(float)
    site = pd.get_dummies(d["site"], dtype=float)  # one column per site = intercept
    X = np.column_stack([d["treatment"].to_numpy(float), site.to_numpy()])
    n, k = X.shape

    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    h = np.einsum("ij,jk,ik->i", X, XtX_inv, X)  # leverage
    a = XtX_inv @ X.T                              # k x n
    var = np.sum(a[0] ** 2 * resid**2 / (1 - h))   # HC2
    se = float(np.sqrt(var))
    est = float(beta[0])
    p = float(2 * stats.t.sf(abs(est / se), n - k))

    with open(args.output, "w") as f:
        json.dump({"estimate": est, "std_error": se, "p_value": p}, f, indent=2)


if __name__ == "__main__":
    main()
