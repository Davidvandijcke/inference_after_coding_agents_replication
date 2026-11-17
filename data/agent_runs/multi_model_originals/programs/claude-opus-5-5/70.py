#!/usr/bin/env python
"""Rerun the chosen Pathways to Work analysis on a dataset with the codebook columns.

Chosen test (selected after exploring data.csv; exploratory, not confirmatory):
    Effect of assignment to Pathways to Work on usual weekly hours worked at
    12 months (hours_m12) among participants at site 1.

Specification (CODEBOOK.md convention, applied within the subgroup):
    OLS of hours_m12 on treatment, site indicators (constant within site 1,
    so absorbed by the intercept) and the baseline covariates age, female,
    parent, education (category indicators), household_size,
    employed_baseline, earnings_baseline, months_unemployed_baseline.
    HC2 standard errors; two-sided t-test of the treatment coefficient.

Usage:
    python score.py --data <path> --output <path>
"""
import argparse
import json

import numpy as np
import pandas as pd
import statsmodels.api as sm

OUTCOME = "hours_m12"
SUBGROUP_SITE = 1
COVARIATES = [
    "age",
    "female",
    "parent",
    "household_size",
    "employed_baseline",
    "earnings_baseline",
    "months_unemployed_baseline",
]


def build_design(df):
    parts = [df[["treatment"]].astype(float)]
    parts.append(pd.get_dummies(df["site"], prefix="site", drop_first=True, dtype=float))
    parts.append(df[COVARIATES].astype(float))
    parts.append(pd.get_dummies(df["education"], prefix="educ", drop_first=True, dtype=float))
    X = pd.concat(parts, axis=1)
    # Drop indicators that are constant in this sample (e.g. site dummies within one site).
    X = X.loc[:, X.nunique() > 1]
    return sm.add_constant(X, has_constant="add")


def run(data_path):
    df = pd.read_csv(data_path)
    df = df[df["site"] == SUBGROUP_SITE].copy()
    if df.empty:
        raise ValueError(f"no rows with site == {SUBGROUP_SITE}")
    X = build_design(df)
    y = df[OUTCOME].astype(float)
    fit = sm.OLS(y, X).fit(cov_type="HC2", use_t=True)
    return {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="CSV with the CODEBOOK.md columns")
    parser.add_argument("--output", required=True, help="path for the JSON result")
    args = parser.parse_args()
    result = run(args.data)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
