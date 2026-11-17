"""Rerun the chosen Pathways to Work analysis on a dataset with the codebook columns.

Chosen test (fixed in advance; no search is performed here):
  Sample:    participants under age 30 at random assignment (age < 30)
  Outcome:   share of the two follow-up surveys at which the participant was
             employed = (employed_m6 + employed_m12) / 2
  Model:     OLS of outcome on treatment + site indicators + baseline covariates
             (age, female, parent, education indicators, household_size,
             employed_baseline, earnings_baseline, months_unemployed_baseline)
  Inference: HC2 robust standard error; two-sided t-test of the treatment
             coefficient using residual degrees of freedom.

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import numpy as np
import pandas as pd
import statsmodels.api as sm

AGE_CUTOFF = 30
COVARIATES = ["age", "female", "parent", "household_size", "employed_baseline",
              "earnings_baseline", "months_unemployed_baseline"]


def design(df):
    site = pd.get_dummies(df["site"].astype("category"), prefix="site", drop_first=True, dtype=float)
    educ = pd.get_dummies(df["education"].astype("category"), prefix="educ", drop_first=True, dtype=float)
    X = pd.concat([df[["treatment"]].astype(float), site, df[COVARIATES].astype(float), educ], axis=1)
    # Drop indicator/covariate columns with no variation in this sample.
    keep = [c for c in X.columns if c == "treatment" or X[c].nunique() > 1]
    return sm.add_constant(X[keep], has_constant="add")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    sub = df[df["age"] < AGE_CUTOFF].copy()
    y = (sub["employed_m6"] + sub["employed_m12"]) / 2.0
    X = design(sub)

    fit = sm.OLS(y.astype(float), X).fit(cov_type="HC2", use_t=True)
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
