"""Rerun the chosen Pathways to Work analysis on a dataset with the codebook's columns.

Chosen analysis (fixed in advance; no search is performed here):
  Outcome:  earnings_m6 (earnings in the month before the 6-month survey, dollars)
  Sample:   all participants
  Model:    OLS of earnings_m6 on treatment + site indicators + baseline covariates
            (age, female, parent, education categories, household_size,
             employed_baseline, earnings_baseline, months_unemployed_baseline)
  Inference: HC2 robust standard error; two-sided t-test of the treatment coefficient
             (residual degrees of freedom).

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf

FORMULA = (
    "earnings_m6 ~ treatment + C(site) + age + female + parent + C(education)"
    " + household_size + employed_baseline + earnings_baseline"
    " + months_unemployed_baseline"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True, help="Input CSV with codebook columns")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    fit = smf.ols(FORMULA, data=df).fit(cov_type="HC2", use_t=True)

    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    with open(args.output, "w") as fh:
        json.dump(result, fh, indent=2)


if __name__ == "__main__":
    main()
