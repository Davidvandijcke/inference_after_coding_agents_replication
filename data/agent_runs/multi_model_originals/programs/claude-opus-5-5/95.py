#!/usr/bin/env python
"""Rerun the chosen Pathways to Work analysis on a dataset with the same columns.

Chosen test (selected after exploration of data.csv; exploratory, not pre-registered):
  Effect of assignment to the program on `debt_collections_m12` (debt in collections
  in the past 6 months, 12-month follow-up), full sample.
  OLS: debt_collections_m12 ~ treatment + site indicators, HC2 robust SE,
  two-sided t-test of the treatment coefficient (codebook convention).

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import numpy as np
import pandas as pd
import statsmodels.api as sm

OUTCOME = "debt_collections_m12"


def estimate(df):
    df = df[[OUTCOME, "treatment", "site"]].dropna()
    site = pd.get_dummies(df["site"].astype("category"), prefix="site", drop_first=True, dtype=float)
    X = pd.concat([df[["treatment"]].astype(float), site], axis=1)
    X = sm.add_constant(X, has_constant="add")
    fit = sm.OLS(df[OUTCOME].astype(float), X).fit(cov_type="HC2")
    return {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    res = estimate(pd.read_csv(a.data))
    with open(a.output, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
