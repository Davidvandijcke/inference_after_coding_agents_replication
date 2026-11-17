"""Rerun the chosen Pathways to Work analysis on a dataset with the data.csv columns.

Chosen analysis (fixed in advance of seeing any new data; no search is performed):
    outcome    employed_m12 (employed in the survey week, 12 months after random assignment)
    sample     all participants
    model      OLS of employed_m12 on treatment, site indicators, and the full set of
               baseline covariates listed in CODEBOOK.md
    inference  HC2 heteroskedasticity-robust standard error, two-sided t-test of a zero
               coefficient on treatment (residual degrees of freedom)

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json
import sys

import numpy as np
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
FORMULA = f"{OUTCOME} ~ treatment + C(site) + " + " + ".join(COVARIATES)


def estimate(df):
    needed = [OUTCOME, "treatment", "site"] + COVARIATES
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    df = df[needed]
    n_all = len(df)
    df = df.dropna()
    if len(df) < n_all:
        print(f"warning: dropped {n_all - len(df)} rows with missing values", file=sys.stderr)
    if set(df["treatment"].unique()) - {0, 1}:
        raise ValueError("treatment must be coded 0/1")

    fit = smf.ols(FORMULA, data=df).fit(cov_type="HC2", use_t=True)
    return {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True, help="CSV with the data.csv columns")
    parser.add_argument("--output", required=True, help="path for the JSON result")
    args = parser.parse_args()

    result = estimate(pd.read_csv(args.data))
    if not all(np.isfinite(v) for v in result.values()):
        raise RuntimeError(f"non-finite result: {result}")
    with open(args.output, "w") as fh:
        json.dump(result, fh, indent=2)
        fh.write("\n")


if __name__ == "__main__":
    main()
