#!/usr/bin/env python3
"""Rerun the single chosen analysis: effect of Pathways to Work on 12-month food
insecurity (food_insecure_m12), OLS of outcome on treatment + site indicators,
HC2 standard errors, two-sided t-test on the treatment coefficient (CODEBOOK.md)."""
import argparse, json
import pandas as pd
import statsmodels.formula.api as smf

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    d = pd.read_csv(a.data)
    m = smf.ols("food_insecure_m12 ~ treatment + C(site)", d).fit(cov_type="HC2")
    out = {"estimate": float(m.params["treatment"]),
           "std_error": float(m.bse["treatment"]),
           "p_value": float(m.pvalues["treatment"])}
    with open(a.output, "w") as f:
        json.dump(out, f, indent=2)

if __name__ == "__main__":
    main()
