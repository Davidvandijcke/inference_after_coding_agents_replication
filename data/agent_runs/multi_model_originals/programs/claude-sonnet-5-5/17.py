#!/usr/bin/env python3
"""Pre-chosen test: does the effect of Pathways to Work on 12-month financial
security (fin_security_m12) differ by baseline employment?

OLS: fin_security_m12 ~ treatment + employed_baseline + treatment:employed_baseline
     + site indicators, HC2 standard errors, two-sided t-test of the
     treatment:employed_baseline coefficient (CODEBOOK analysis convention).
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
    r = smf.ols("fin_security_m12 ~ treatment * employed_baseline + C(site)", d).fit(
        cov_type="HC2", use_t=True)
    k = "treatment:employed_baseline"
    out = {"estimate": float(r.params[k]), "std_error": float(r.bse[k]),
           "p_value": float(r.pvalues[k])}
    with open(a.output, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
