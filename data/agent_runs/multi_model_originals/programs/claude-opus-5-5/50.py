"""Pathways to Work: confirmatory rerun of the chosen analysis.

Estimand: effect of assignment to the program on employment at the 12-month
follow-up (employed_m12), full sample.
Model: OLS of employed_m12 on treatment, site indicators, and a fixed set of
baseline covariates (age, female, parent, education indicators, household_size,
employed_baseline, earnings_baseline, months_unemployed_baseline).
Inference: HC2 robust standard error, two-sided t-test (df = n - k) of the
treatment coefficient.

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats

OUTCOME = "employed_m12"
CONTINUOUS = ["age", "female", "parent", "household_size", "employed_baseline",
              "earnings_baseline", "months_unemployed_baseline"]
CATEGORICAL = ["site", "education"]


def design(df):
    parts = [pd.Series(1.0, index=df.index, name="const"),
             df["treatment"].astype(float).rename("treatment")]
    for c in CATEGORICAL:
        parts.append(pd.get_dummies(df[c], prefix=c, drop_first=True, dtype=float))
    parts.append(df[CONTINUOUS].astype(float))
    X = pd.concat(parts, axis=1)
    # drop columns that are constant (e.g. a dummy with no variation in a new dataset)
    keep = [c for c in X.columns if c == "const" or X[c].nunique() > 1]
    return X[keep]


def hc2_ols(y, X, term):
    Xv, yv = X.to_numpy(), y.to_numpy(dtype=float)
    n, k = Xv.shape
    if np.linalg.matrix_rank(Xv) < k:
        raise ValueError("design matrix is rank deficient")
    XtX_inv = np.linalg.inv(Xv.T @ Xv)
    beta = XtX_inv @ Xv.T @ yv
    resid = yv - Xv @ beta
    h = np.einsum("ij,jk,ik->i", Xv, XtX_inv, Xv)
    meat = (Xv * (resid ** 2 / (1.0 - h))[:, None]).T @ Xv
    V = XtX_inv @ meat @ XtX_inv
    j = list(X.columns).index(term)
    est, se = float(beta[j]), float(np.sqrt(V[j, j]))
    df_resid = n - k
    t = est / se
    p = float(2 * stats.t.sf(abs(t), df_resid))
    return est, se, p, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    df = pd.read_csv(a.data)
    cols = [OUTCOME, "treatment"] + CATEGORICAL + CONTINUOUS
    df = df.dropna(subset=cols)
    est, se, p, n = hc2_ols(df[OUTCOME], design(df), "treatment")
    with open(a.output, "w") as f:
        json.dump({"estimate": est, "std_error": se, "p_value": p,
                   "outcome": OUTCOME, "n": n}, f, indent=2)


if __name__ == "__main__":
    main()
