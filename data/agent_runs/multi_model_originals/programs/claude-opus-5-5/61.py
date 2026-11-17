#!/usr/bin/env python
"""Rerun the chosen Pathways to Work analysis on a dataset with the same columns.

Chosen analysis (fixed in advance of any new data; no search is performed here):
  Outcome: an overall well-being summary index, i.e. the equally weighted mean of eight
           outcomes at both follow-ups (16 components). Each component is standardized
           by the control-group mean and SD in the supplied data and signed so that
           higher values are better:
             + employed, hours, earnings, fin_security, health
             - food_insecure, debt_collections, distress
           (training is excluded because its desirable direction is ambiguous.)
  Model:   OLS of the index on treatment + site indicators + baseline covariates
           (age, female, parent, education indicators, household_size,
           employed_baseline, earnings_baseline, months_unemployed_baseline).
  Inference: HC2 robust standard error; two-sided t-test of the treatment coefficient
           with n - rank(X) degrees of freedom.

Usage: python score.py --data <path.csv> --output <path.json>
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats

SIGNS = {
    "employed": 1, "hours": 1, "earnings": 1, "fin_security": 1, "health": 1,
    "food_insecure": -1, "debt_collections": -1, "distress": -1,
}
TIMEPOINTS = ("m6", "m12")
NUMERIC_COVARIATES = [
    "age", "female", "parent", "household_size", "employed_baseline",
    "earnings_baseline", "months_unemployed_baseline",
]


def build_index(df):
    control = df["treatment"] == 0
    components = []
    for t in TIMEPOINTS:
        for outcome, sign in SIGNS.items():
            v = df[f"{outcome}_{t}"].astype(float)
            sd = v[control].std(ddof=1)
            if not np.isfinite(sd) or sd == 0:
                continue  # a constant component carries no information
            components.append(sign * (v - v[control].mean()) / sd)
    if not components:
        raise ValueError("No index component has variation in the control group.")
    return pd.concat(components, axis=1).mean(axis=1)


def design_matrix(df):
    parts = [
        pd.Series(1.0, index=df.index, name="const"),
        df["treatment"].astype(float),
        df[NUMERIC_COVARIATES].astype(float),
        pd.get_dummies(df["site"].astype(int), prefix="site", drop_first=True, dtype=float),
        pd.get_dummies(df["education"].astype(int), prefix="edu", drop_first=True, dtype=float),
    ]
    return pd.concat(parts, axis=1)


def ols_hc2(y, X):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    resid = y - X @ beta
    leverage = np.einsum("ij,jk,ik->i", X, xtx_inv, X)
    omega = resid ** 2 / (1.0 - leverage)
    vcov = xtx_inv @ (X.T * omega) @ X @ xtx_inv
    rank = np.linalg.matrix_rank(X)
    return beta, vcov, len(y) - rank


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    needed = (["treatment", "site", "education"] + NUMERIC_COVARIATES
              + [f"{o}_{t}" for t in TIMEPOINTS for o in SIGNS])
    df = pd.read_csv(args.data)
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df = df.dropna(subset=needed).reset_index(drop=True)

    y = build_index(df)
    X = design_matrix(df)
    beta, vcov, dof = ols_hc2(y, X)
    j = list(X.columns).index("treatment")
    estimate = float(beta[j])
    std_error = float(np.sqrt(vcov[j, j]))
    p_value = float(2 * stats.t.sf(abs(estimate / std_error), dof))

    result = {"estimate": estimate, "std_error": std_error, "p_value": p_value}
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
