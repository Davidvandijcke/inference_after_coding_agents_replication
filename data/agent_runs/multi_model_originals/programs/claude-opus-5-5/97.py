"""Rerun the chosen Pathways to Work analysis on a dataset with the codebook's columns.

Chosen test: effect of assignment to the program on 6-month earnings among
participants under age 30 at random assignment. OLS of earnings_m6 on treatment
and site indicators within the subgroup, HC2 robust standard errors, two-sided
t-test of the treatment coefficient (residual degrees of freedom).

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf

OUTCOME = "earnings_m6"
AGE_CUTOFF = 30  # subgroup: age < 30


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="CSV with the codebook columns")
    parser.add_argument("--output", required=True, help="Path for the JSON result")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    sub = df.loc[df["age"] < AGE_CUTOFF, [OUTCOME, "treatment", "site"]].dropna()

    fit = smf.ols(f"{OUTCOME} ~ treatment + C(site)", data=sub).fit(cov_type="HC2", use_t=True)
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
