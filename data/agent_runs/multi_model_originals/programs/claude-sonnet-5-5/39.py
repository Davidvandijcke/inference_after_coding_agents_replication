#!/usr/bin/env python3
"""Pathways to Work: effect of treatment on employment at 6 months.

OLS of employed_m6 on treatment + site indicators, HC2 standard errors,
two-sided t-test of the treatment coefficient (CODEBOOK.md convention).
Usage: python score.py --data <path> --output <path>
"""
import argparse, json
import pandas as pd
import statsmodels.formula.api as smf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    d = pd.read_csv(a.data)
    r = smf.ols("employed_m6 ~ treatment + C(site)", d).fit(cov_type="HC2")
    out = {"estimate": float(r.params["treatment"]),
           "std_error": float(r.bse["treatment"]),
           "p_value": float(r.pvalues["treatment"])}
    with open(a.output, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
