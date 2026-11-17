"""Rerun the chosen Pathways to Work analysis on a dataset with the same columns.

Chosen test (fixed in advance; no search is repeated here):
  Outcome:   employed_m12 (employed in the survey week, 12 months after random assignment)
  Sample:    all participants
  Model:     OLS of employed_m12 on treatment + site indicators + baseline covariates
             (age, female, parent, education, household_size, employed_baseline,
              earnings_baseline, months_unemployed_baseline)
  Inference: HC2 robust standard error, two-sided t-test of the treatment coefficient
             (residual degrees of freedom), per the CODEBOOK.md analysis convention.

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
    parser.add_argument("--data", required=True, help="CSV with the codebook columns")
    parser.add_argument("--output", required=True, help="path for the JSON result")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    needed = [OUTCOME, "treatment", "site"] + COVARIATES
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise SystemExit(f"missing required columns: {missing}")
    df = df[needed].dropna()

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
