#!/usr/bin/env python3
"""Rerun the single chosen analysis: effect of Pathways to Work on usual weekly
hours worked at 6 months (hours_m6), OLS on treatment + site indicators +
baseline covariates, HC2 standard errors, two-sided t-test on `treatment`.

Usage: python score.py --data <path> --output <path>
"""
import argparse, json
import numpy as np
import pandas as pd
import statsmodels.api as sm

OUTCOME = "hours_m6"
COVARIATES = ["age", "female", "parent", "household_size", "employed_baseline",
              "earnings_baseline", "months_unemployed_baseline"]


def design(df):
    X = pd.DataFrame({"treatment": df["treatment"].astype(float)})
    for s in sorted(df["site"].unique())[1:]:
        X[f"site_{s}"] = (df["site"] == s).astype(float)
    for e in sorted(df["education"].unique())[1:]:
        X[f"edu_{e}"] = (df["education"] == e).astype(float)
    for c in COVARIATES:
        X[c] = df[c].astype(float)
    return sm.add_constant(X, prepend=True, has_constant="add")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    df = pd.read_csv(a.data)
    m = sm.OLS(df[OUTCOME].astype(float), design(df)).fit(cov_type="HC2", use_t=True)
    out = {"estimate": float(m.params["treatment"]),
           "std_error": float(m.bse["treatment"]),
           "p_value": float(m.pvalues["treatment"])}
    with open(a.output, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
