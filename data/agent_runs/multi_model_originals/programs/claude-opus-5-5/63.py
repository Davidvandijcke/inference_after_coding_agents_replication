"""Rerun the chosen Pathways to Work analysis on a dataset with the codebook columns.

Chosen test (selected in exploratory analysis of the original data): effect of
assignment to the program on enrollment in education/job training at 12 months
(training_m12) among men (female == 0).  OLS of training_m12 on treatment, site
indicators and baseline covariates, HC2 robust standard errors, two-sided t-test
of the treatment coefficient.

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import numpy as np
import pandas as pd
import statsmodels.api as sm

OUTCOME = "training_m12"
COVARIATES = ["age", "parent", "education", "household_size", "employed_baseline",
              "earnings_baseline", "months_unemployed_baseline"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    d = pd.read_csv(args.data)
    men = d[d["female"] == 0].copy()

    sites = pd.get_dummies(men["site"].astype(int), prefix="site", drop_first=True, dtype=float)
    X = pd.concat([men[["treatment"]].astype(float), sites, men[COVARIATES].astype(float)], axis=1)
    X = sm.add_constant(X, has_constant="add")
    # Drop constant covariates (possible in small samples); treatment is always kept.
    X = X[[c for c in X.columns if c in ("const", "treatment") or X[c].std() > 0]]

    fit = sm.OLS(men[OUTCOME].astype(float), X).fit(cov_type="HC2", use_t=True)
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
