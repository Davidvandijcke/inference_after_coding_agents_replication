"""Pathways to Work: effect of program assignment on employment at 12 months.

Chosen analysis (single pre-specified test, no search):
  OLS of employed_m12 on treatment + site indicators + baseline covariates
  (age, female, parent, education, household_size, employed_baseline,
  earnings_baseline, months_unemployed_baseline), full sample,
  HC2 robust standard error, two-sided t-test of the treatment coefficient.

Usage: python score.py --data <path.csv> --output <path.json>
"""
import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf

OUTCOME = "employed_m12"
COVARIATES = [
    "age", "female", "parent", "education", "household_size",
    "employed_baseline", "earnings_baseline", "months_unemployed_baseline",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    needed = [OUTCOME, "treatment", "site"] + COVARIATES
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
