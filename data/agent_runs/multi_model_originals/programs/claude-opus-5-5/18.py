"""Rerun the chosen Pathways to Work analysis on a dataset with the codebook columns.

Chosen test: effect of assignment to the program on employment at 12 months
(employed_m12). OLS of the outcome on treatment, site indicators and the full
set of baseline covariates. HC2 standard errors, two-sided t-test of the
treatment coefficient.

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf

FORMULA = (
    "employed_m12 ~ treatment + C(site)"
    " + age + female + parent + C(education) + household_size"
    " + employed_baseline + earnings_baseline + months_unemployed_baseline"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="CSV with the codebook columns")
    parser.add_argument("--output", required=True, help="Path for the JSON result")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    fit = smf.ols(FORMULA, data=df).fit(cov_type="HC2", use_t=True)
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
