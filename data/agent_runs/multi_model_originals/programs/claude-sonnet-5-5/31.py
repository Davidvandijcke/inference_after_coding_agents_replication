#!/usr/bin/env python
"""Re-run the single pre-chosen Pathways to Work analysis on another dataset.

Chosen analysis (exploratory lead from the original data; this script does not search):
  outcome   : debt_collections_m6
  subgroup  : education >= 3 (some college or more)
  model     : OLS of outcome on treatment + site indicators, HC2 robust SE,
              two-sided t-test of the treatment coefficient (CODEBOOK convention)

Usage: python score.py --data <path> --output <path>
Output JSON: {"estimate", "std_error", "p_value"} (plus n for reference).
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
    a = ap.parse_args()

    d = pd.read_csv(a.data)
    d = d[d["education"] >= 3]
    y = d[OUTCOME].to_numpy(float)
    t = d["treatment"].to_numpy(float)
    # treatment + full set of site dummies (acts as the intercept)
    site = pd.get_dummies(d["site"]).to_numpy(float)
    X = np.column_stack([t, site])

    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    h = np.einsum("ij,jk,ik->i", X, XtX_inv, X)          # leverage
    meat = (X * (resid**2 / (1 - h))[:, None]).T @ X     # HC2
    V = XtX_inv @ meat @ XtX_inv
    est, se = float(beta[0]), float(np.sqrt(V[0, 0]))
    df = len(y) - np.linalg.matrix_rank(X)
    p = float(2 * stats.t.sf(abs(est / se), df))

    with open(a.output, "w") as f:
        json.dump({"estimate": est, "std_error": se, "p_value": p, "n": int(len(y))}, f, indent=2)


if __name__ == "__main__":
    main()
