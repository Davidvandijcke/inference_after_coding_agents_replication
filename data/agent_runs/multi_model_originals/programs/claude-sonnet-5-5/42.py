#!/usr/bin/env python
"""Pathways to Work: effect of treatment on employment at 12 months.

Chosen analysis (per CODEBOOK.md convention): OLS of employed_m12 on treatment
and site indicators, HC2 robust standard errors, two-sided t-test of the
treatment coefficient. Runs this single test only; no searching.

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf

OUTCOME = "employed_m12"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    fit = smf.ols(f"{OUTCOME} ~ treatment + C(site)", df).fit(cov_type="HC2", use_t=True)
    out = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
