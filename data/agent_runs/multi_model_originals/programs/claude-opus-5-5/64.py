"""Rerun the chosen Pathways to Work analysis on a dataset with the CODEBOOK.md columns.

Chosen analysis (fixed in advance; no search is done here):
  Outcome:   employed_m12 (employed in the survey week, 12 months after random assignment)
  Sample:    all participants
  Model:     OLS of employed_m12 on treatment, site indicators and baseline covariates
             (age, female, parent, education indicators, household_size,
             employed_baseline, earnings_baseline, months_unemployed_baseline)
  Inference: HC2 robust standard error, two-sided t-test of a zero treatment coefficient
             (df = n - number of regressors)

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats

OUTCOME = "employed_m12"
CONTINUOUS_COVARIATES = [
    "age",
    "female",
    "parent",
    "household_size",
    "employed_baseline",
    "earnings_baseline",
    "months_unemployed_baseline",
]


def design_matrix(df):
    site = pd.get_dummies(df["site"].astype("category"), prefix="site", drop_first=True)
    educ = pd.get_dummies(df["education"].astype("category"), prefix="educ", drop_first=True)
    X = pd.concat(
        [
            pd.Series(1.0, index=df.index, name="const"),
            df["treatment"].astype(float),
            site.astype(float),
            df[CONTINUOUS_COVARIATES].astype(float),
            educ.astype(float),
        ],
        axis=1,
    )
    # Drop indicator columns that are constant zero (e.g. an empty category).
    return X.loc[:, (X != 0).any(axis=0)]


def hc2_ols(y, X, coef):
    Xv = X.to_numpy()
    n, k = Xv.shape
    if np.linalg.matrix_rank(Xv) < k:
        raise ValueError("Design matrix is rank-deficient; check the input data.")
    xtx_inv = np.linalg.inv(Xv.T @ Xv)
    beta = xtx_inv @ Xv.T @ y
    resid = y - Xv @ beta
    leverage = np.einsum("ij,jk,ik->i", Xv, xtx_inv, Xv)
    meat = (Xv * (resid**2 / (1.0 - leverage))[:, None]).T @ Xv
    vcov = xtx_inv @ meat @ xtx_inv
    j = list(X.columns).index(coef)
    est = float(beta[j])
    se = float(np.sqrt(vcov[j, j]))
    df_resid = n - k
    p = float(2 * stats.t.sf(abs(est / se), df_resid))
    return est, se, p, n, df_resid


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    cols = ["site", "treatment", "education", OUTCOME] + CONTINUOUS_COVARIATES
    df = df.dropna(subset=cols)
    X = design_matrix(df)
    y = df[OUTCOME].astype(float).to_numpy()
    est, se, p, n, df_resid = hc2_ols(y, X, "treatment")

    result = {
        "estimate": est,
        "std_error": se,
        "p_value": p,
        "outcome": OUTCOME,
        "sample": "all participants",
        "model": "OLS: outcome ~ treatment + site indicators + baseline covariates; HC2 SE; two-sided t-test",
        "n": int(n),
        "df_resid": int(df_resid),
    }
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
