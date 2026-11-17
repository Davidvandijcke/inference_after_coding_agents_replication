"""Pathways to Work: effect of program assignment on employment at 12 months.

Runs one pre-chosen analysis (no search): OLS of employed_m12 on treatment and
site indicators, full sample, HC2 robust standard errors, two-sided t-test of
the treatment coefficient (residual degrees of freedom).

Usage: python score.py --data <path to csv> --output <path to json>
"""
import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf

OUTCOME = "employed_m12"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="CSV with the codebook columns")
    parser.add_argument("--output", required=True, help="where to write the JSON result")
    args = parser.parse_args()

    df = pd.read_csv(args.data, usecols=[OUTCOME, "treatment", "site"]).dropna()
    model = smf.ols(f"{OUTCOME} ~ treatment + C(site)", data=df).fit(
        cov_type="HC2", use_t=True
    )

    result = {
        "estimate": float(model.params["treatment"]),
        "std_error": float(model.bse["treatment"]),
        "p_value": float(model.pvalues["treatment"]),
    }
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
