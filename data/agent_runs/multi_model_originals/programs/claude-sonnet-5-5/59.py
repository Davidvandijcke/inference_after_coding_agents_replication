#!/usr/bin/env python
"""Rerun the single chosen analysis: effect of Pathways to Work on training
enrollment at 6 months (training_m6), full sample.

OLS of training_m6 on treatment + site indicators, HC2 standard errors,
two-sided t-test of the treatment coefficient (codebook convention).
Usage: python score.py --data <path> --output <path>
"""
import argparse, json
import numpy as np
import pandas as pd
from scipy import stats

OUTCOME = "training_m6"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    d = pd.read_csv(a.data)
    X = pd.concat([d[["treatment"]].astype(float),
                   pd.get_dummies(d["site"], prefix="site").astype(float)], axis=1)
    X = X.to_numpy()
    y = d[OUTCOME].to_numpy(dtype=float)
    n, k = X.shape

    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    h = np.einsum("ij,jk,ik->i", X, XtX_inv, X)       # leverage
    meat = X.T @ (X * (resid**2 / (1 - h))[:, None])  # HC2
    cov = XtX_inv @ meat @ XtX_inv
    est, se = float(beta[0]), float(np.sqrt(cov[0, 0]))
    p = float(2 * stats.t.sf(abs(est / se), df=n - k))

    with open(a.output, "w") as f:
        json.dump({"estimate": est, "std_error": se, "p_value": p}, f, indent=2)


if __name__ == "__main__":
    main()
