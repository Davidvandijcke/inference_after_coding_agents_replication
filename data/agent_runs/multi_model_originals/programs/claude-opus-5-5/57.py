"""Rerun the chosen Pathways to Work analysis on a dataset with the same columns.

Chosen test: intent-to-treat effect of assignment to the program on employment
at the 6-month follow-up (employed_m6), full sample. OLS of employed_m6 on
treatment, site indicators and the eight baseline covariates; HC2 robust
standard errors; two-sided t-test of a zero treatment coefficient.

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf

OUTCOME = "employed_m6"
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to input CSV")
    parser.add_argument("--output", required=True, help="Path to output JSON")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    cols = [OUTCOME, "treatment", "site"] + COVARIATES
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")
    df = df[cols].dropna()

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
