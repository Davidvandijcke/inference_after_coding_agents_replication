"""Pathways to Work: program effect on 12-month earnings (full sample).

OLS of earnings_m12 on treatment, site indicators and baseline covariates,
HC2 robust standard errors, two-sided t-test of the treatment coefficient
(the analysis convention in CODEBOOK.md).

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf

FORMULA = (
    "earnings_m12 ~ treatment + C(site) + age + female + parent + C(education)"
    " + household_size + employed_baseline + earnings_baseline"
    " + months_unemployed_baseline"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

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
