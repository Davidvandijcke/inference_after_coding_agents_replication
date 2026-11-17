"""Rerun the chosen Pathways to Work analysis on a dataset with the codebook columns.

Chosen test (selected after exploration of data.csv; exploratory, not pre-registered):
    Effect of assignment to the program on psychological distress (Kessler-6)
    at the 6-month follow-up, full sample.
    OLS: distress_m6 ~ treatment + site indicators; HC2 robust SE;
    two-sided t-test of the treatment coefficient (df = residual df).

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf

OUTCOME = "distress_m6"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="CSV with the codebook columns")
    ap.add_argument("--output", required=True, help="Path for the JSON result")
    args = ap.parse_args()

    df = pd.read_csv(args.data, usecols=[OUTCOME, "treatment", "site"]).dropna()
    fit = smf.ols(f"{OUTCOME} ~ treatment + C(site)", data=df).fit(cov_type="HC2", use_t=True)

    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
