"""Rerun the chosen Pathways to Work analysis on a dataset with the codebook columns.

Chosen test: effect of assignment to the program on monthly earnings at the
12-month follow-up (earnings_m12). OLS of the outcome on treatment, site
indicators, and the eight baseline covariates; HC2 standard errors; two-sided
t-test of the treatment coefficient.

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf

OUTCOME = "earnings_m12"
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
    parser.add_argument("--output", required=True, help="Path for the JSON result")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    needed = [OUTCOME, "treatment", "site"] + COVARIATES
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns: {missing}")
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
