"""Rerun the chosen Pathways to Work analysis on a dataset with the codebook's columns.

Chosen analysis (selected after exploration of data.csv; see REPORT.md):
  Effect of assignment to the program on `debt_collections_m12` (debt in
  collections in the past 6 months, measured at the 12-month follow-up), full
  sample. OLS of the outcome on `treatment`, site indicators and the baseline
  covariates; HC2 standard errors; two-sided t-test of the treatment coefficient.

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf

OUTCOME = "debt_collections_m12"
FORMULA = (
    f"{OUTCOME} ~ treatment + C(site) + age + female + parent + C(education)"
    " + household_size + employed_baseline + earnings_baseline"
    " + months_unemployed_baseline"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True, help="CSV with the codebook columns")
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
