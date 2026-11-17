"""Rerun the chosen Pathways to Work analysis on a dataset with the codebook columns.

Chosen test: intent-to-treat effect of assignment on earnings 12 months after
random assignment (earnings_m12), full sample. OLS of the outcome on the
treatment indicator, site indicators, and pre-specified baseline covariates,
with HC2 standard errors and a two-sided t-test of the treatment coefficient.

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf

OUTCOME = "earnings_m12"
FORMULA = (
    f"{OUTCOME} ~ treatment + C(site) + age + female + parent + C(education)"
    " + household_size + employed_baseline + earnings_baseline"
    " + months_unemployed_baseline"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="CSV with codebook columns")
    parser.add_argument("--output", required=True, help="Path for JSON results")
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
