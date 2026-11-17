"""Rerun the single chosen analysis: effect of Pathways to Work on training
enrollment at 12 months (training_m12), full sample.

OLS of training_m12 on treatment + site indicators, HC2 standard errors,
two-sided t-test of the treatment coefficient (CODEBOOK.md convention).

Usage: python score.py --data <path> --output <path>
"""
import argparse, json
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
    y = d[OUTCOME].to_numpy(float)
    sites = pd.get_dummies(d["site"], prefix="s", drop_first=True).to_numpy(float)
    X = np.column_stack([np.ones(len(d)), d["treatment"].to_numpy(float), sites])
    n, k = X.shape

    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    h = np.einsum("ij,jk,ik->i", X, XtX_inv, X)          # leverage
    meat = (X * (resid**2 / (1 - h))[:, None]).T @ X     # HC2
    V = XtX_inv @ meat @ XtX_inv

    est, se = float(beta[1]), float(np.sqrt(V[1, 1]))
    p = float(2 * stats.t.sf(abs(est / se), df=n - k))   # t reference, as in statsmodels HC2
    with open(a.output, "w") as f:
        json.dump({"estimate": est, "std_error": se, "p_value": p}, f, indent=2)


if __name__ == "__main__":
    main()
