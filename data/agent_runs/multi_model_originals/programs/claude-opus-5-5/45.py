#!/usr/bin/env python
"""Rerun the single chosen Pathways to Work analysis on a dataset with the same columns.

Chosen test (fixed in advance; no search is performed here):
  Subgroup: participants employed at random assignment (employed_baseline == 1).
  Outcome:  employed_m6 (employed in the survey week, 6 months after assignment).
  Model:    OLS of employed_m6 on treatment + site indicators (randomization was within site).
  Inference: HC2 heteroskedasticity-robust standard error; two-sided t-test of the
             treatment coefficient = 0, with residual degrees of freedom (n - k).

Usage: python score.py --data <path> --output <path>
Writes a JSON object {"estimate": ..., "std_error": ..., "p_value": ...}.
"""
import argparse
import json
import sys

import numpy as np
import pandas as pd
from scipy import stats

SUBGROUP_COL = "employed_baseline"
SUBGROUP_VALUE = 1
OUTCOME = "employed_m6"
TREATMENT = "treatment"
SITE = "site"


def hc2_treatment_test(df):
    y = df[OUTCOME].to_numpy(dtype=float)
    t = df[TREATMENT].to_numpy(dtype=float)
    sites = sorted(df[SITE].unique())
    # Intercept + treatment + site indicators (first site is the reference category).
    cols = [np.ones(len(df)), t] + [(df[SITE] == s).to_numpy(dtype=float) for s in sites[1:]]
    X = np.column_stack(cols)
    n, k = X.shape
    if np.linalg.matrix_rank(X) < k:
        raise ValueError("Design matrix is rank deficient (e.g. no treatment variation in the subgroup).")
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    lev = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
    # HC2 weights e_i^2 / (1 - h_ii); observations with leverage 1 have zero residual and contribute nothing.
    denom = 1.0 - lev
    w = np.where(denom > 1e-12, resid**2 / np.where(denom > 1e-12, denom, 1.0), 0.0)
    meat = (X * w[:, None]).T @ X
    vcov = XtX_inv @ meat @ XtX_inv
    est = float(beta[1])
    se = float(np.sqrt(vcov[1, 1]))
    df_resid = n - k
    p = float(2 * stats.t.sf(abs(est / se), df_resid))
    return est, se, p, n, df_resid


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="CSV with the Pathways to Work columns")
    ap.add_argument("--output", required=True, help="Path for the JSON result")
    args = ap.parse_args()

    data = pd.read_csv(args.data)
    needed = [SUBGROUP_COL, OUTCOME, TREATMENT, SITE]
    missing = [c for c in needed if c not in data.columns]
    if missing:
        sys.exit(f"Missing required columns: {missing}")
    for c in [SUBGROUP_COL, OUTCOME, TREATMENT]:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    n_all = len(data)
    data = data.dropna(subset=needed)
    if len(data) < n_all:
        print(f"Note: dropped {n_all - len(data)} rows with missing values in {needed}", file=sys.stderr)

    sub = data[data[SUBGROUP_COL] == SUBGROUP_VALUE]
    est, se, p, n, df_resid = hc2_treatment_test(sub)

    result = {"estimate": est, "std_error": se, "p_value": p}
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"n={n}, df={df_resid}: estimate={est:.6f}, std_error={se:.6f}, p_value={p:.6f}", file=sys.stderr)


if __name__ == "__main__":
    main()
