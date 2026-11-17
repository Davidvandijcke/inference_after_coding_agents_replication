"""Rerun the chosen Pathways to Work analysis on a dataset with the same columns.

Chosen test (fixed in advance of any new data; no search is repeated here):
    Outcome:   food_insecure_m12 (household food insecurity, 12-month follow-up)
    Sample:    all participants
    Model:     OLS of food_insecure_m12 on treatment + site indicators (no covariates)
    Inference: HC2 robust standard error, two-sided t-test of the treatment coefficient

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf

OUTCOME = "food_insecure_m12"
REQUIRED = [OUTCOME, "treatment", "site"]


def score(df):
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    df = df[REQUIRED].dropna()
    fit = smf.ols(f"{OUTCOME} ~ treatment + C(site)", data=df).fit(cov_type="HC2", use_t=True)
    return {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="CSV with the Pathways to Work columns")
    parser.add_argument("--output", required=True, help="path for the JSON result")
    args = parser.parse_args()
    result = score(pd.read_csv(args.data))
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
