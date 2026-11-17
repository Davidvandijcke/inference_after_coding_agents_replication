#!/usr/bin/env python
"""Pathways to Work: chosen analysis only.

Effect of random assignment to the program on employment 12 months after
random assignment (employed_m12), full sample. OLS of employed_m12 on
treatment, site indicators, and pre-specified baseline covariates; HC2
standard errors; two-sided t-test of the treatment coefficient.

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf

OUTCOME = "employed_m12"
COVARIATES = [
    "age", "female", "parent", "C(education)", "household_size",
    "employed_baseline", "earnings_baseline", "months_unemployed_baseline",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    d = pd.read_csv(args.data)
    terms = ["treatment"]
    if d["site"].nunique() > 1:
        terms.append("C(site)")
    terms += COVARIATES
    fit = smf.ols(f"{OUTCOME} ~ " + " + ".join(terms), data=d).fit(cov_type="HC2", use_t=True)

    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
