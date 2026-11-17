"""Pathways to Work: rerun the chosen analysis on a dataset with the data.csv columns.

Chosen analysis (fixed; no search is performed here):
  Effect of assignment to the program on self-rated health at 6 months (health_m6),
  full sample, OLS of health_m6 on treatment + site indicators + baseline covariates,
  HC2 standard errors, two-sided t-test of the treatment coefficient.

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf

OUTCOME = "health_m6"
COVARIATES = [
    "age", "female", "parent", "education", "household_size",
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
