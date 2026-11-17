#!/usr/bin/env python3
"""Re-run the single pre-chosen analysis: effect of Pathways to Work on
self-rated health at 12 months (health_m12) among participants at site 3.

Convention (CODEBOOK.md): OLS of the outcome on treatment plus site indicators,
HC2 robust standard errors, two-sided t-test of the treatment coefficient,
applied within the subgroup. Within a single site the site indicators collapse
into the intercept. No search is repeated here.

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats

SITE = 3
OUTCOME = "health_m12"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    d = pd.read_csv(a.data)
    d = d[d["site"] == SITE]
    y = d[OUTCOME].to_numpy(float)
    X = np.column_stack([d["treatment"].to_numpy(float), np.ones(len(d))])
    n, k = X.shape

    XtXi = np.linalg.inv(X.T @ X)
    beta = XtXi @ X.T @ y
    resid = y - X @ beta
    h = np.einsum("ij,jk,ik->i", X, XtXi, X)
    meat = (X * (resid**2 / (1.0 - h))[:, None]).T @ X  # HC2
    cov = XtXi @ meat @ XtXi
    est = float(beta[0])
    se = float(np.sqrt(cov[0, 0]))
    p = float(2 * stats.t.sf(abs(est / se), df=n - k))

    with open(a.output, "w") as f:
        json.dump({"estimate": est, "std_error": se, "p_value": p}, f, indent=2)


if __name__ == "__main__":
    main()
