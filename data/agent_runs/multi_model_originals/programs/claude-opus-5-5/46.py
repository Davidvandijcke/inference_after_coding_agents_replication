"""Rerun the chosen Pathways to Work analysis on a dataset with the same columns.

Chosen test (fixed in advance of any new data; no search is repeated here):
    Outcome:   earnings_m6 (earnings in the previous month, 6-month follow-up)
    Sample:    all participants
    Model:     OLS of earnings_m6 on treatment, site indicators, and the eight
               baseline covariates listed in BASELINE
    Inference: HC2 heteroskedasticity-robust standard error, two-sided t-test
               of a zero coefficient on treatment (df = n - rank of design)

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats

OUTCOME = "earnings_m6"
BASELINE = [
    "age",
    "female",
    "parent",
    "education",
    "household_size",
    "employed_baseline",
    "earnings_baseline",
    "months_unemployed_baseline",
]


def design_matrix(df):
    # Treatment first, then a full set of site indicators (they absorb the intercept).
    sites = pd.get_dummies(df["site"], prefix="site", dtype=float)
    X = pd.concat([df[["treatment"]].astype(float), sites, df[BASELINE].astype(float)], axis=1)
    return X.to_numpy()


def hc2_treatment_test(X, y):
    n = X.shape[0]
    rank = np.linalg.matrix_rank(X)
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    leverage = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
    omega = resid**2 / (1.0 - leverage)
    bread_row = (X @ XtX_inv)[:, 0]
    se = float(np.sqrt(np.sum(bread_row**2 * omega)))
    est = float(beta[0])
    t_stat = est / se
    p = float(2 * stats.t.sf(abs(t_stat), n - rank))
    return est, se, p


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="CSV with the Pathways to Work columns")
    parser.add_argument("--output", required=True, help="Path for the JSON result")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    cols = ["site", "treatment", OUTCOME] + BASELINE
    df = df[cols].dropna()

    est, se, p = hc2_treatment_test(design_matrix(df), df[OUTCOME].to_numpy(dtype=float))
    result = {"estimate": est, "std_error": se, "p_value": p}
    with open(args.output, "w") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
