"""Rerun the chosen Pathways to Work analysis on a dataset with the data.csv columns.

Chosen test: effect of assignment to the program on the financial security
index averaged over the 6- and 12-month follow-ups,
    fin_security_avg = (fin_security_m6 + fin_security_m12) / 2,
estimated by OLS on `treatment`, site indicators and all baseline covariates
(education entered as categories), with HC2 standard errors and a two-sided
t-test of the `treatment` coefficient (residual degrees of freedom).

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf

FORMULA = (
    "fin_security_avg ~ treatment + C(site) + age + female + parent"
    " + C(education) + household_size + employed_baseline"
    " + earnings_baseline + months_unemployed_baseline"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True, help="input CSV path")
    parser.add_argument("--output", required=True, help="output JSON path")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    df["fin_security_avg"] = (df["fin_security_m6"] + df["fin_security_m12"]) / 2

    fit = smf.ols(FORMULA, data=df).fit(cov_type="HC2", use_t=True)
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    with open(args.output, "w") as fh:
        json.dump(result, fh, indent=2)


if __name__ == "__main__":
    main()
