"""Pathways to Work: effect of program assignment on 6-month food insecurity.

Chosen analysis (fixed in advance; no search is performed here):
  OLS of food_insecure_m6 on treatment + site indicators, full sample,
  HC2 robust standard errors, two-sided t-test of the treatment coefficient.

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    d = pd.read_csv(args.data)
    fit = smf.ols("food_insecure_m6 ~ treatment + C(site)", data=d).fit(cov_type="HC2", use_t=True)
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
