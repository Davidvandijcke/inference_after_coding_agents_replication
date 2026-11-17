"""Rerun the chosen Pathways to Work analysis on a dataset with the same columns.

Chosen test (fixed in advance of applying to new data; no search is performed):
  Outcome:   employed_m12 (employed in the survey week, 12 months after assignment)
  Model:     OLS of employed_m12 on treatment + site indicators + baseline covariates
  Inference: HC2 heteroskedasticity-robust SE, two-sided t-test of the treatment coefficient

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf

OUTCOME = "employed_m12"
COVARIATES = [
    "age",
    "female",
    "parent",
    "education",
    "household_size",
    "employed_baseline",
    "earnings_baseline",
    "months_unemployed_baseline",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True, help="Path to input CSV")
    parser.add_argument("--output", required=True, help="Path to output JSON")
    args = parser.parse_args()

    cols = [OUTCOME, "treatment", "site"] + COVARIATES
    df = pd.read_csv(args.data, usecols=cols).dropna()

    formula = f"{OUTCOME} ~ treatment + C(site) + " + " + ".join(COVARIATES)
    model = smf.ols(formula, data=df).fit(cov_type="HC2", use_t=True)

    result = {
        "estimate": float(model.params["treatment"]),
        "std_error": float(model.bse["treatment"]),
        "p_value": float(model.pvalues["treatment"]),
    }
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
