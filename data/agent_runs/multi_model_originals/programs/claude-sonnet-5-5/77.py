#!/usr/bin/env python3
"""Pathways to Work: effect of treatment on 12-month earnings (earnings_m12).

Convention from CODEBOOK.md: OLS of the outcome on `treatment` plus site
indicators, HC2 robust standard errors, two-sided t-test on `treatment`.
Runs only this single pre-fixed analysis; no outcome/subgroup search.

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf

OUTCOME = "earnings_m12"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    m = smf.ols(f"{OUTCOME} ~ treatment + C(site)", df).fit(cov_type="HC2")
    out = {
        "estimate": float(m.params["treatment"]),
        "std_error": float(m.bse["treatment"]),
        "p_value": float(m.pvalues["treatment"]),
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
