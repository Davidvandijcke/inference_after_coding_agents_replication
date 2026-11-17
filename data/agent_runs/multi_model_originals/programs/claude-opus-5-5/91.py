"""Rerun the chosen Pathways to Work analysis on a dataset with the same columns.

Chosen analysis (fixed in advance of applying to new data; no search):
  Full sample, outcome = employed_m12 (employed in the survey week, 12 months
  after random assignment). OLS of the outcome on the treatment indicator, site
  indicators, and all eight baseline covariates from CODEBOOK.md, entered
  linearly. HC2 standard errors; two-sided t-test of the treatment coefficient
  (residual degrees of freedom).

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf

OUTCOME = "employed_m12"
COVARIATES = [
    "age",
    "female",
    "parent",
    "education",
    "household_size",
    "employed_baseline",
    "earnings_baseline",
    "months_unemployed_baseline",
]


def estimate(df):
    cols = [OUTCOME, "treatment", "site"] + COVARIATES
    df = df[cols].dropna()
    formula = f"{OUTCOME} ~ treatment + C(site) + " + " + ".join(COVARIATES)
    fit = smf.ols(formula, data=df).fit(cov_type="HC2", use_t=True)
    return {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True, help="CSV with the codebook columns")
    parser.add_argument("--output", required=True, help="path for the JSON result")
    args = parser.parse_args()

    result = estimate(pd.read_csv(args.data))
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
