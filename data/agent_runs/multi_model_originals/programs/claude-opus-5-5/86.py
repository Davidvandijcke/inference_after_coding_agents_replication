"""Pathways to Work: rerun the chosen analysis on a dataset with the data.csv columns.

Chosen analysis (fixed, no search): effect of random assignment to the program
on employment 12 months after random assignment (employed_m12), full sample.
OLS of employed_m12 on treatment and site indicators, HC2 robust standard
errors, two-sided t-test of the treatment coefficient (CODEBOOK.md convention).

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    d = pd.read_csv(args.data, usecols=["employed_m12", "treatment", "site"]).dropna()
    fit = smf.ols("employed_m12 ~ treatment + C(site)", data=d).fit(cov_type="HC2", use_t=True)
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
