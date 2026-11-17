"""Rerun the chosen Pathways to Work analysis on a dataset with the same columns.

Chosen test: effect of assignment to the program (`treatment`) on employment
in the 6-month survey week (`employed_m6`), full sample. Per the CODEBOOK
convention: OLS of the outcome on `treatment`, site indicators and the baseline
covariates; HC2 robust standard errors; two-sided t-test of a zero coefficient.

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf

OUTCOME = "employed_m6"
FORMULA = (
    f"{OUTCOME} ~ treatment + C(site)"
    " + age + female + parent + C(education) + household_size"
    " + employed_baseline + earnings_baseline + months_unemployed_baseline"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="input CSV path")
    parser.add_argument("--output", required=True, help="output JSON path")
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
