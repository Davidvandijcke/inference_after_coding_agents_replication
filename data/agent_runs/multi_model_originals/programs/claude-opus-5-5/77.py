"""Rerun the chosen Pathways to Work analysis on a dataset with the CODEBOOK.md columns.

Chosen analysis: intent-to-treat effect on earnings at 12 months (earnings_m12),
full sample, OLS on treatment + site indicators + all baseline covariates,
HC2 standard errors, two-sided t-test of the treatment coefficient.

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf

FORMULA = (
    "earnings_m12 ~ treatment + C(site)"
    " + age + female + parent + C(education) + household_size"
    " + employed_baseline + earnings_baseline + months_unemployed_baseline"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True, help="CSV with the CODEBOOK.md columns")
    parser.add_argument("--output", required=True, help="path for the JSON result")
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
