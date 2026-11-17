#!/usr/bin/env python
"""Rerun the single chosen analysis: effect of Pathways to Work on 6-month
earnings among participants who live with a child under 18 (parent == 1).

OLS of earnings_m6 on treatment + site indicators, HC2 robust SEs, two-sided
t-test on the treatment coefficient (CODEBOOK convention).

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
    s = d[d["parent"] == 1]
    r = smf.ols("earnings_m6 ~ treatment + C(site)", s).fit(cov_type="HC2", use_t=True)
    out = {
        "estimate": float(r.params["treatment"]),
        "std_error": float(r.bse["treatment"]),
        "p_value": float(r.pvalues["treatment"]),
    }
    with open(a.output, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
