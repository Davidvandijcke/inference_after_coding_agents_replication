#!/usr/bin/env python
"""Rerun the single chosen Pathways to Work analysis on another dataset.

Chosen analysis: effect of treatment on the financial security index
(0-20), averaged over the 6- and 12-month follow-ups, estimated per the
CODEBOOK convention: OLS on treatment + site indicators, HC2 robust
standard errors, two-sided t-test of the treatment coefficient.
No covariates, no subgroups, no search.

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    d = pd.read_csv(args.data)
    d["fin_security_avg"] = (d["fin_security_m6"] + d["fin_security_m12"]) / 2
    res = smf.ols("fin_security_avg ~ treatment + C(site)", d).fit(
        cov_type="HC2", use_t=True
    )
    out = {
        "estimate": float(res.params["treatment"]),
        "std_error": float(res.bse["treatment"]),
        "p_value": float(res.pvalues["treatment"]),
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
