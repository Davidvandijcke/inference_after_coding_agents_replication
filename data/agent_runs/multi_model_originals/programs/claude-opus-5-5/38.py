#!/usr/bin/env python
"""Pathways to Work: effect of program assignment on 12-month employment.

Chosen analysis (single pre-specified test, no search):
  OLS of employed_m12 on treatment + site indicators + baseline covariates
  (age, female, parent, education categories, household_size,
  employed_baseline, earnings_baseline, months_unemployed_baseline),
  full sample, HC2 robust standard error, two-sided t-test of the
  treatment coefficient (df = n - k).

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats

OUTCOME = "employed_m12"
NUMERIC_COVARIATES = ["age", "female", "parent", "household_size", "employed_baseline",
                      "earnings_baseline", "months_unemployed_baseline"]


def design(df):
    site = pd.get_dummies(df["site"].astype(int), prefix="site", drop_first=True, dtype=float)
    educ = pd.get_dummies(df["education"].astype(int), prefix="educ", drop_first=True, dtype=float)
    X = pd.concat([pd.Series(1.0, index=df.index, name="const"),
                   df["treatment"].astype(float), site, educ,
                   df[NUMERIC_COVARIATES].astype(float)], axis=1)
    # Drop columns that are constant (other than the intercept), e.g. a single site.
    keep = [c for c in X.columns if c in ("const", "treatment") or X[c].nunique() > 1]
    return X[keep]


def estimate(df):
    cols = ["treatment", "site", "education", OUTCOME] + NUMERIC_COVARIATES
    df = df[cols].dropna()
    X = design(df).to_numpy()
    y = df[OUTCOME].astype(float).to_numpy()
    n, k = X.shape
    if np.linalg.matrix_rank(X) < k:
        raise ValueError("Design matrix is rank-deficient; cannot estimate the treatment effect.")
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    leverage = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
    meat = (X * (resid ** 2 / (1.0 - leverage))[:, None]).T @ X  # HC2
    vcov = XtX_inv @ meat @ XtX_inv
    j = 1  # treatment column
    est, se = float(beta[j]), float(np.sqrt(vcov[j, j]))
    t = est / se
    p = float(2 * stats.t.sf(abs(t), df=n - k))
    return {"estimate": est, "std_error": se, "p_value": p}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    res = estimate(pd.read_csv(args.data))
    with open(args.output, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
