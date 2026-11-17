#!/usr/bin/env python3
"""Rerun the single chosen analysis: effect of Pathways to Work on usual weekly
hours at 12 months (hours_m12), full sample, per the CODEBOOK convention:
OLS of outcome on treatment + site indicators, HC2 robust SE, two-sided t-test.
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
    df = pd.read_csv(a.data)
    r = smf.ols("hours_m12 ~ treatment + C(site)", df).fit(cov_type="HC2", use_t=True)
    out = {"estimate": float(r.params["treatment"]),
           "std_error": float(r.bse["treatment"]),
           "p_value": float(r.pvalues["treatment"])}
    with open(a.output, "w") as f:
        json.dump(out, f, indent=2)

if __name__ == "__main__":
    main()
