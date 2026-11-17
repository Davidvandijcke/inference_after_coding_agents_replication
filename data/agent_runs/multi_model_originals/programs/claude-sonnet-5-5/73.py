#!/usr/bin/env python3
"""Rerun the single chosen analysis: does the Pathways to Work effect on financial
security (mean of fin_security_m6 and fin_security_m12) differ for participants
living with a child under 18 (parent=1) versus not (parent=0)?

Model: OLS of outcome on treatment, parent, treatment x parent and site indicators.
HC2 standard errors, two-sided t-test on the treatment:parent coefficient
(CODEBOOK.md convention). No search is performed.

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
    d["fs_avg"] = (d["fin_security_m6"] + d["fin_security_m12"]) / 2
    m = smf.ols("fs_avg ~ treatment * parent + C(site)", d).fit(cov_type="HC2", use_t=True)
    k = "treatment:parent"
    out = {
        "estimate": float(m.params[k]),
        "std_error": float(m.bse[k]),
        "p_value": float(m.pvalues[k]),
    }
    with open(a.output, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
