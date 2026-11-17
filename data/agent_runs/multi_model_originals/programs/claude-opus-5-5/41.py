"""Pathways to Work: pooled 12-month earnings effect (pre-specified single test).

OLS of earnings_m12 on treatment, site indicators and a fixed set of baseline
covariates; HC2 robust standard error; two-sided t-test of the treatment
coefficient (residual degrees of freedom), per the CODEBOOK.md convention.

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf

OUTCOME = "earnings_m12"
COVARIATES = [
    "age", "female", "parent", "C(education)", "household_size",
    "employed_baseline", "earnings_baseline", "months_unemployed_baseline",
]
RAW_COLUMNS = [OUTCOME, "treatment", "site", "age", "female", "parent", "education",
               "household_size", "employed_baseline", "earnings_baseline",
               "months_unemployed_baseline"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"missing required columns: {missing}")
    df = df[RAW_COLUMNS].dropna()

    formula = f"{OUTCOME} ~ treatment + C(site) + " + " + ".join(COVARIATES)
    fit = smf.ols(formula, data=df).fit(cov_type="HC2", use_t=True)

    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    with open(args.output, "w") as fh:
        json.dump(result, fh, indent=2)


if __name__ == "__main__":
    main()
