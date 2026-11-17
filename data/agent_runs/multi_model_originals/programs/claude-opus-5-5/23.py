#!/usr/bin/env python
"""Rerun the chosen Pathways to Work analysis on a dataset with the CODEBOOK.md columns.

Chosen test: effect of random assignment (`treatment`) on monthly earnings at the
12-month follow-up (`earnings_m12`), full sample. OLS of earnings_m12 on treatment,
site indicators, and the pre-specified baseline covariates (age, female, parent,
education indicators, household_size, employed_baseline, earnings_baseline,
months_unemployed_baseline). HC2 robust standard error; two-sided t-test of a zero
treatment coefficient (df = n - number of regressors).

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats

OUTCOME = "earnings_m12"
NUMERIC_COVARIATES = ["age", "female", "parent", "household_size", "employed_baseline",
                      "earnings_baseline", "months_unemployed_baseline"]
CATEGORICAL_COVARIATES = ["site", "education"]


def design_matrix(df):
    parts = [pd.Series(1.0, index=df.index, name="const"),
             df["treatment"].astype(float)]
    for col in CATEGORICAL_COVARIATES:
        dummies = pd.get_dummies(df[col].astype(int), prefix=col, drop_first=True, dtype=float)
        parts.append(dummies)
    parts.append(df[NUMERIC_COVARIATES].astype(float))
    return pd.concat(parts, axis=1)


def ols_hc2(y, X):
    Xv = X.to_numpy(dtype=float)
    yv = y.to_numpy(dtype=float)
    XtX_inv = np.linalg.inv(Xv.T @ Xv)
    beta = XtX_inv @ Xv.T @ yv
    resid = yv - Xv @ beta
    leverage = np.einsum("ij,jk,ik->i", Xv, XtX_inv, Xv)
    meat = Xv.T @ (Xv * (resid ** 2 / (1.0 - leverage))[:, None])
    vcov = XtX_inv @ meat @ XtX_inv
    j = list(X.columns).index("treatment")
    est = float(beta[j])
    se = float(np.sqrt(vcov[j, j]))
    df_resid = Xv.shape[0] - Xv.shape[1]
    p = float(2 * stats.t.sf(abs(est / se), df_resid))
    return est, se, p, int(Xv.shape[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    cols = ["treatment", OUTCOME] + CATEGORICAL_COVARIATES + NUMERIC_COVARIATES
    df = df[cols].dropna()
    X = design_matrix(df)
    est, se, p, n = ols_hc2(df[OUTCOME], X)

    result = {"estimate": est, "std_error": se, "p_value": p,
              "outcome": OUTCOME, "sample": "full", "n": n,
              "method": "OLS on treatment + site FE + baseline covariates; HC2 SE; two-sided t-test"}
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
