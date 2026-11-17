#!/usr/bin/env python
"""Rerun the chosen Pathways to Work analysis on a dataset with the same columns.

Chosen test: effect of random assignment to the program on enrollment in
education or job training at the 12-month follow-up (training_m12), full sample.
OLS of training_m12 on treatment plus site indicators, HC2 robust standard
errors, two-sided t-test of a zero coefficient on treatment.

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf

OUTCOME = "training_m12"
REQUIRED = ["site", "treatment", OUTCOME]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True, help="CSV with the evaluation columns")
    parser.add_argument("--output", required=True, help="path for the JSON result")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise SystemExit(f"missing required columns: {missing}")
    df = df[REQUIRED].dropna()

    fit = smf.ols(f"{OUTCOME} ~ treatment + C(site)", data=df).fit(
        cov_type="HC2", use_t=True
    )
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
