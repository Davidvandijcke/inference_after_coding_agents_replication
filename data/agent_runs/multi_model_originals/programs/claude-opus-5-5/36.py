#!/usr/bin/env python
"""Pathways to Work: rerun the single chosen analysis on a dataset with the same columns.

Chosen test (fixed in advance of any new data; no search is performed here):
  Outcome   : employed_m12 (employed in the survey week, 12 months after random assignment)
  Sample    : all participants
  Model     : OLS of employed_m12 on treatment + site indicators + baseline covariates
              (age, female, parent, education indicators, household_size,
               employed_baseline, earnings_baseline, months_unemployed_baseline)
  Inference : HC2 heteroskedasticity-robust SE; two-sided t-test of the treatment
              coefficient = 0 with residual degrees of freedom (n - k).

Usage: python score.py --data <path.csv> --output <path.json>
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats

OUTCOME = "employed_m12"
NUMERIC_COVARIATES = [
    "age", "female", "parent", "household_size", "employed_baseline",
    "earnings_baseline", "months_unemployed_baseline",
]
CATEGORICAL = ["site", "education"]


def dummies(series):
    """Indicator columns for all observed levels except the lowest (reference)."""
    levels = sorted(series.unique())
    return np.column_stack([(series == lv).astype(float) for lv in levels[1:]]) if len(levels) > 1 \
        else np.empty((len(series), 0))


def drop_collinear(X, keep_first=2, tol=1e-10):
    """Drop columns that are linearly dependent on earlier ones (never intercept/treatment)."""
    kept = []
    for j in range(X.shape[1]):
        cand = kept + [j]
        if np.linalg.matrix_rank(X[:, cand], tol=tol * max(1.0, np.abs(X[:, cand]).max())) == len(cand):
            kept.append(j)
        elif j < keep_first:
            raise ValueError("treatment indicator is collinear with site/covariates; cannot estimate")
    return X[:, kept]


def estimate(df):
    cols = [OUTCOME, "treatment"] + CATEGORICAL + NUMERIC_COVARIATES
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    df = df[cols].apply(pd.to_numeric, errors="coerce").dropna()

    y = df[OUTCOME].to_numpy(float)
    X = np.column_stack([
        np.ones(len(df)),
        df["treatment"].to_numpy(float),
        dummies(df["site"]),
        dummies(df["education"]),
        df[NUMERIC_COVARIATES].to_numpy(float),
    ])
    X = drop_collinear(X)
    n, k = X.shape
    if n <= k:
        raise ValueError("not enough observations for the model")

    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    leverage = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
    omega = resid ** 2 / (1.0 - leverage)                     # HC2 weights
    vcov = XtX_inv @ (X.T * omega) @ X @ XtX_inv

    est = float(beta[1])
    se = float(np.sqrt(vcov[1, 1]))
    df_resid = n - k
    p = float(2 * stats.t.sf(abs(est / se), df_resid))
    return {"estimate": est, "std_error": se, "p_value": p,
            "outcome": OUTCOME, "n": int(n), "df_resid": int(df_resid)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    res = estimate(pd.read_csv(a.data))
    with open(a.output, "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
