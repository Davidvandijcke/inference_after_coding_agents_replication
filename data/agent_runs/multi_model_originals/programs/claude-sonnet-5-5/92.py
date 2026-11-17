"""Pathways to Work: chosen exploratory analysis (no search is repeated).

Effect of treatment on food insecurity at 6 months (food_insecure_m6) among
male participants (female == 0): OLS on treatment + site indicators, HC2
standard errors, two-sided t-test on the treatment coefficient, per CODEBOOK.md.

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats


def hc2_treatment_effect(y, treat, site):
    dummies = pd.get_dummies(site).astype(float)  # full set of site indicators = intercept
    X = np.column_stack([treat.astype(float), dummies.values])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    lever = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
    u2 = resid**2 / (1.0 - lever)
    cov = XtX_inv @ (X.T * u2) @ X @ XtX_inv
    est = beta[0]
    se = float(np.sqrt(cov[0, 0]))
    df = len(y) - X.shape[1]
    p = float(2 * stats.t.sf(abs(est / se), df))
    return float(est), se, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    d = pd.read_csv(args.data)
    sub = d[d["female"] == 0]
    est, se, p = hc2_treatment_effect(
        sub["food_insecure_m6"].to_numpy(float), sub["treatment"].to_numpy(), sub["site"].to_numpy()
    )
    with open(args.output, "w") as f:
        json.dump({"estimate": est, "std_error": se, "p_value": p}, f, indent=2)


if __name__ == "__main__":
    main()
