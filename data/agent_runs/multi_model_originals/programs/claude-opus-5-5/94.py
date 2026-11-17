#!/usr/bin/env python
"""Rerun the chosen Pathways to Work analysis on a dataset with the codebook's columns.

Chosen test (selected during exploratory analysis; this script does not repeat the search):
  Subgroup: long-term unemployed at random assignment (months_unemployed_baseline >= 12).
  Outcome:  employed_m12 (employed in the survey week at the 12-month follow-up).
  Model:    OLS of employed_m12 on treatment + site indicators, within the subgroup.
  Inference: HC2 robust standard error; two-sided t-test of the treatment coefficient
             (t distribution with residual degrees of freedom).

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats

SUBGROUP_MIN_MONTHS = 12
OUTCOME = "employed_m12"


def hc2_treatment_test(y, treatment, site):
    sites = np.unique(site)
    X = np.column_stack(
        [treatment, np.ones(len(y))] + [(site == s).astype(float) for s in sites[1:]]
    )
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    leverage = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
    meat = X.T @ (X * (resid**2 / (1.0 - leverage))[:, None])
    cov = XtX_inv @ meat @ XtX_inv
    estimate = float(beta[0])
    std_error = float(np.sqrt(cov[0, 0]))
    df_resid = len(y) - X.shape[1]
    p_value = float(2 * stats.t.sf(abs(estimate / std_error), df_resid))
    return estimate, std_error, p_value


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True, help="CSV with the codebook columns")
    parser.add_argument("--output", required=True, help="Path for the JSON result")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    sub = df[df["months_unemployed_baseline"] >= SUBGROUP_MIN_MONTHS]
    sub = sub[["treatment", "site", OUTCOME]].dropna()

    estimate, std_error, p_value = hc2_treatment_test(
        sub[OUTCOME].to_numpy(float),
        sub["treatment"].to_numpy(float),
        sub["site"].to_numpy(),
    )
    with open(args.output, "w") as f:
        json.dump({"estimate": estimate, "std_error": std_error, "p_value": p_value}, f, indent=2)


if __name__ == "__main__":
    main()
