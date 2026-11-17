"""Rerun the single chosen analysis: effect of Pathways to Work on 6-month
psychological distress (distress_m6) among participants living with a child
under 18 (parent == 1).

Convention (CODEBOOK.md): OLS of the outcome on `treatment` plus site
indicators, HC2 robust standard errors, two-sided t-test on `treatment`.

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf


def run(df):
    sub = df[df["parent"] == 1]
    fit = smf.ols("distress_m6 ~ treatment + C(site)", data=sub).fit(
        cov_type="HC2", use_t=True
    )
    return {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    res = run(pd.read_csv(args.data))
    with open(args.output, "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
