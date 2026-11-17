#!/usr/bin/env python
"""Chosen analysis: ITT effect of Pathways to Work on training_m6 (enrolled in
education/job training at 6 months), full sample.
OLS of training_m6 on treatment + site indicators, HC2 SEs, two-sided t-test."""
import argparse, json
import pandas as pd
import statsmodels.formula.api as smf

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    d = pd.read_csv(a.data)
    m = smf.ols("training_m6 ~ treatment + C(site)", d).fit(cov_type="HC2", use_t=True)
    res = {"estimate": float(m.params["treatment"]),
           "std_error": float(m.bse["treatment"]),
           "p_value": float(m.pvalues["treatment"])}
    with open(a.output, "w") as f:
        json.dump(res, f, indent=2)

if __name__ == "__main__":
    main()
