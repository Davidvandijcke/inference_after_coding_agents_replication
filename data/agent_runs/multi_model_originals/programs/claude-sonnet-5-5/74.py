#!/usr/bin/env python
"""Pathways to Work: pooled effect of assignment to the program on employment
12 months after random assignment (employed_m12).

OLS of employed_m12 on treatment + site indicators, HC2 standard errors,
two-sided t-test on the treatment coefficient (CODEBOOK.md convention).
Runs only this one pre-fixed analysis; no searching over outcomes or subgroups.

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
    a = ap.parse_args()

    d = pd.read_csv(a.data)
    res = smf.ols(f"{OUTCOME} ~ treatment + C(site)", d).fit(cov_type="HC2")
    out = {
        "estimate": float(res.params["treatment"]),
        "std_error": float(res.bse["treatment"]),
        "p_value": float(res.pvalues["treatment"]),
    }
    with open(a.output, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
