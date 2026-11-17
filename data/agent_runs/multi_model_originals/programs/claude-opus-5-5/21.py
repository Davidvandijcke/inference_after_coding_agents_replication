#!/usr/bin/env python
"""Pathways to Work: rerun the single pre-chosen analysis.

Chosen test: effect of assignment to Pathways to Work on enrollment in
education or job training at the 6-month follow-up (`training_m6`).

Specification (CODEBOOK.md analysis convention):
  OLS of training_m6 on treatment + site indicators + baseline covariates
  (age, female, parent, education indicators, household_size,
  employed_baseline, earnings_baseline, months_unemployed_baseline),
  HC2 robust standard errors, two-sided t-test (df = n - k) of the
  treatment coefficient.

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats

OUTCOME = "training_m6"
NUMERIC_COVARIATES = [
    "age", "female", "parent", "household_size", "employed_baseline",
    "earnings_baseline", "months_unemployed_baseline",
]
CATEGORICAL_COVARIATES = ["site", "education"]


def design_matrix(df):
    parts = [pd.Series(1.0, index=df.index, name="const"),
             df["treatment"].astype(float)]
    for col in CATEGORICAL_COVARIATES:
        parts.append(pd.get_dummies(df[col], prefix=col, drop_first=True, dtype=float))
    parts.append(df[NUMERIC_COVARIATES].astype(float))
    X = pd.concat(parts, axis=1)
    # Drop any all-constant dummy columns (e.g. a category absent from the data).
    keep = [c for c in X.columns if c == "const" or X[c].nunique() > 1]
    return X[keep]


def ols_hc2(y, X):
    Xv = X.to_numpy(dtype=float)
    yv = np.asarray(y, dtype=float)
    n, k = Xv.shape
    XtX_inv = np.linalg.pinv(Xv.T @ Xv)
    beta = XtX_inv @ Xv.T @ yv
    resid = yv - Xv @ beta
    leverage = np.einsum("ij,jk,ik->i", Xv, XtX_inv, Xv)
    omega = resid ** 2 / (1.0 - leverage)
    meat = (Xv * omega[:, None]).T @ Xv
    vcov = XtX_inv @ meat @ XtX_inv
    rank = np.linalg.matrix_rank(Xv)
    return beta, vcov, n - rank


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    needed = ["treatment", OUTCOME] + CATEGORICAL_COVARIATES + NUMERIC_COVARIATES
    df = df.dropna(subset=needed)

    X = design_matrix(df)
    beta, vcov, dof = ols_hc2(df[OUTCOME], X)
    j = list(X.columns).index("treatment")
    est = float(beta[j])
    se = float(np.sqrt(vcov[j, j]))
    t = est / se
    p = float(2 * stats.t.sf(abs(t), dof))

    result = {
        "estimate": est,
        "std_error": se,
        "p_value": p,
        "outcome": OUTCOME,
        "specification": "OLS of training_m6 on treatment + site FE + baseline covariates; HC2 SE; two-sided t-test",
        "n": int(len(df)),
        "df_resid": int(dof),
    }
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
