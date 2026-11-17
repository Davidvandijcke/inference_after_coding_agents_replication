#!/usr/bin/env python3
"""Reruns the single chosen analysis (no search):
effect of Pathways to Work on employment at 12 months (employed_m12) among
participants with a bachelor's degree or higher (education == 4).
OLS of outcome on treatment + site indicators, HC2 SEs, two-sided t-test on treatment.
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
    s = d[d["education"] == 4]
    fit = smf.ols("employed_m12 ~ treatment + C(site)", data=s).fit(cov_type="HC2", use_t=True)
    out = {"estimate": float(fit.params["treatment"]),
           "std_error": float(fit.bse["treatment"]),
           "p_value": float(fit.pvalues["treatment"])}
    with open(a.output, "w") as f:
        json.dump(out, f, indent=2)

if __name__ == "__main__":
    main()
