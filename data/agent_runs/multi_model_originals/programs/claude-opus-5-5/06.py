#!/usr/bin/env python
"""Pathways to Work: pre-specified rerun of the chosen analysis.

Effect of random assignment to the program on employment 12 months after
random assignment (employed_m12), full sample. OLS of employed_m12 on the
treatment indicator, site indicators (randomization strata) and baseline
covariates; HC2 standard errors; two-sided t-test of the treatment coefficient.

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
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
