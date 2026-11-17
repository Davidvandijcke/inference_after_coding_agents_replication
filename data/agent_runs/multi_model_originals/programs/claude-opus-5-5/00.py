"""Rerun the chosen Pathways to Work analysis on a dataset with the codebook's columns.

Chosen test: effect of random assignment to the program on employment at the
12-month follow-up (employed_m12), full sample. OLS of employed_m12 on treatment,
site indicators and the baseline covariates listed below; HC2 standard errors;
two-sided t-test of a zero treatment coefficient (residual degrees of freedom).

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf

OUTCOME = "employed_m12"
FORMULA = (
    f"{OUTCOME} ~ treatment + C(site) + age + female + parent + C(education)"
    " + household_size + employed_baseline + earnings_baseline"
    " + months_unemployed_baseline"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True, help="CSV with codebook columns")
    parser.add_argument("--output", required=True, help="path for the JSON result")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    fit = smf.ols(FORMULA, data=data).fit(cov_type="HC2", use_t=True)

    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
