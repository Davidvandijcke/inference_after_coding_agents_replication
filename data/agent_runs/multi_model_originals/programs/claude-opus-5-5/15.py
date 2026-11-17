"""Pathways to Work: overall program effect on employment at 12 months.

Chosen analysis (no search): OLS of employed_m12 on treatment and site
indicators, HC2 robust standard errors, two-sided t-test of the treatment
coefficient.

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

    d = pd.read_csv(args.data, usecols=["site", "treatment", "employed_m12"]).dropna()
    m = smf.ols("employed_m12 ~ treatment + C(site)", data=d).fit(cov_type="HC2", use_t=True)

    out = {
        "estimate": float(m.params["treatment"]),
        "std_error": float(m.bse["treatment"]),
        "p_value": float(m.pvalues["treatment"]),
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
