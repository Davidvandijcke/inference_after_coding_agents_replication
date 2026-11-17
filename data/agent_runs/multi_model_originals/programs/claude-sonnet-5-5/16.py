#!/usr/bin/env python
"""Rerun the single chosen analysis: effect of Pathways to Work on 12-month
earnings among participants aged 37 or older (the median age in the original
data; fixed here so it is not re-derived from new data).

Convention (CODEBOOK.md): OLS of earnings_m12 on treatment + site indicators,
HC2 robust standard errors, two-sided t-test on the treatment coefficient,
estimated within the subgroup.

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf

AGE_CUTOFF = 37


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    d = pd.read_csv(args.data)
    s = d[d["age"] >= AGE_CUTOFF]
    fit = smf.ols("earnings_m12 ~ treatment + C(site)", data=s).fit(cov_type="HC2", use_t=True)
    out = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
