"""Pathways to Work: rerun the single chosen analysis.

Effect of assignment to the program on employment at the 12-month follow-up
(employed_m12), full sample. OLS of the outcome on treatment, site indicators,
and the baseline covariates, with HC2 standard errors and a two-sided t-test
of a zero treatment coefficient (the CODEBOOK.md analysis convention).

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf

OUTCOME = "employed_m12"
COVARIATES = [
    "age", "female", "parent", "education", "household_size",
    "employed_baseline", "earnings_baseline", "months_unemployed_baseline",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True, help="CSV with the codebook columns")
    parser.add_argument("--output", required=True, help="path for the JSON result")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    cols = [OUTCOME, "treatment", "site"] + COVARIATES
    df = df[cols].dropna()

    formula = f"{OUTCOME} ~ treatment + C(site) + " + " + ".join(COVARIATES)
    fit = smf.ols(formula, data=df).fit(cov_type="HC2", use_t=True)

    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
