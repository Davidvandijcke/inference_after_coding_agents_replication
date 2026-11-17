"""Pathways to Work: pre-specified effect of program assignment on 12-month employment.

OLS of employed_m12 on treatment, site fixed effects, and all baseline covariates,
with HC2 standard errors and a two-sided t-test of the treatment coefficient
(the CODEBOOK.md analysis convention). Full sample; no subgroup or outcome search.

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf

FORMULA = (
    "employed_m12 ~ treatment + C(site)"
    " + age + female + parent + C(education) + household_size"
    " + employed_baseline + earnings_baseline + months_unemployed_baseline"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    fit = smf.ols(FORMULA, data=df).fit(cov_type="HC2", use_t=True)
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
