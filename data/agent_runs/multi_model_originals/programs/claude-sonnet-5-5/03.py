"""Rerun the single chosen analysis: effect of Pathways to Work on the 12-month
financial security index (fin_security_m12) among participants at site 1.

Convention (CODEBOOK.md): OLS of the outcome on `treatment` plus site indicators
(constant within one site, so absorbed by the intercept), HC2 robust standard
errors, two-sided t-test of a zero treatment coefficient.

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf

OUTCOME = "fin_security_m12"
SITE = 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    sub = df[df["site"] == SITE]
    fit = smf.ols(f"{OUTCOME} ~ treatment", data=sub).fit(cov_type="HC2", use_t=True)
    out = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
