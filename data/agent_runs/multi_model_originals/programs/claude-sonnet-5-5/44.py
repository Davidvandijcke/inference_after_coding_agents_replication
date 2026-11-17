#!/usr/bin/env python
"""Re-run the single chosen analysis from the Pathways to Work exploration.

Chosen test: effect of treatment on 12-month food insecurity (food_insecure_m12)
among participants living in households of 3 or more people (household_size >= 3).
Estimation follows CODEBOOK.md: OLS of the outcome on treatment plus site
indicators, HC2 standard errors, two-sided t-test on the treatment coefficient.
No search over outcomes, follow-ups or subgroups is performed here.

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf

OUTCOME = "food_insecure_m12"
MIN_HOUSEHOLD_SIZE = 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    sub = df[df["household_size"] >= MIN_HOUSEHOLD_SIZE]
    fit = smf.ols(f"{OUTCOME} ~ treatment + C(site)", data=sub).fit(cov_type="HC2")
    res = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    with open(args.output, "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
