"""Rerun the chosen Pathways to Work analysis on a dataset with the codebook's columns.

Chosen analysis (fixed in advance, no search): effect of random assignment on
earnings in the month before the 12-month follow-up (`earnings_m12`), full sample.
OLS of earnings_m12 on treatment and site indicators; HC2 robust standard error;
two-sided t-test of a zero treatment coefficient (residual degrees of freedom).

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf

OUTCOME = "earnings_m12"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="CSV with the codebook's columns")
    parser.add_argument("--output", required=True, help="path for the JSON result")
    args = parser.parse_args()

    df = pd.read_csv(args.data, usecols=[OUTCOME, "treatment", "site"]).dropna()
    fit = smf.ols(f"{OUTCOME} ~ treatment + C(site)", data=df).fit(cov_type="HC2", use_t=True)

    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
