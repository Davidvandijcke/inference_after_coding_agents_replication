#!/usr/bin/env python
"""Pathways to Work: re-run the chosen exploratory test on a dataset with the same columns.

Test: does the program effect on the 6-month financial security index differ for
parents vs. non-parents? OLS of fin_security_m6 on treatment x parent, site x parent
indicators, and baseline covariates; HC2 robust SEs; two-sided t-test of the
treatment:parent coefficient (parent effect minus non-parent effect).

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf

FORMULA = (
    "fin_security_m6 ~ treatment * parent + C(site) * parent"
    " + age + female + C(education) + household_size + employed_baseline"
    " + earnings_baseline + months_unemployed_baseline"
)
TERM = "treatment:parent"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True, help="CSV with the Pathways to Work columns")
    ap.add_argument("--output", required=True, help="path for the JSON result")
    args = ap.parse_args()

    d = pd.read_csv(args.data)
    fit = smf.ols(FORMULA, data=d).fit(cov_type="HC2", use_t=True)
    result = {
        "estimate": float(fit.params[TERM]),
        "std_error": float(fit.bse[TERM]),
        "p_value": float(fit.pvalues[TERM]),
    }
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
