#!/usr/bin/env python
"""Rerun the chosen analysis: treatment x age interaction on 6-month financial security.

OLS of fin_security_m6 on treatment, age (in decades, centered at the sample mean),
treatment x age and site indicators; HC2 standard errors; two-sided t-test.
Reported coefficient: treatment:age (change in treatment effect per 10 years of age).
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
    d["age10"] = (d["age"] - d["age"].mean()) / 10
    r = smf.ols("fin_security_m6 ~ treatment * age10 + C(site)", d).fit(cov_type="HC2")
    k = "treatment:age10"
    out = {"estimate": float(r.params[k]), "std_error": float(r.bse[k]), "p_value": float(r.pvalues[k])}
    with open(a.output, "w") as f:
        json.dump(out, f, indent=2)

if __name__ == "__main__":
    main()
