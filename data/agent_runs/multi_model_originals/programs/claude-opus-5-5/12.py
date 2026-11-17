"""Pathways to Work: effect of program assignment on employment at 12 months.

Pre-chosen single test (no search): OLS of employed_m12 on treatment and site
indicators, HC2 robust standard errors, two-sided t-test of the treatment
coefficient (analysis convention in CODEBOOK.md).

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf

OUTCOME = "employed_m12"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True, help="CSV with the Pathways to Work columns")
    parser.add_argument("--output", required=True, help="Path for the JSON result")
    args = parser.parse_args()

    d = pd.read_csv(args.data, usecols=[OUTCOME, "treatment", "site"]).dropna()
    fit = smf.ols(f"{OUTCOME} ~ treatment + C(site)", data=d).fit(cov_type="HC2", use_t=True)

    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
