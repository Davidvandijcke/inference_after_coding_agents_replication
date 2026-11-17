"""Rerun the chosen Pathways to Work analysis on a dataset with the data.csv columns.

Chosen test: full-sample intent-to-treat effect on earnings in the month before the
12-month survey (earnings_m12). Following CODEBOOK.md, earnings_m12 is regressed by OLS on
treatment plus site indicators, with HC2 standard errors and a two-sided t-test on treatment.
No search over outcomes, follow-ups or subgroups is done here.

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf

OUTCOME = "earnings_m12"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", required=True, help="CSV with the data.csv columns")
    ap.add_argument("--output", required=True, help="path for the JSON result")
    args = ap.parse_args()

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
