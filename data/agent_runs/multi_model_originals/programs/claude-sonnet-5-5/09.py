"""Rerun the single chosen analysis: effect of Pathways to Work on 12-month earnings.

Model (CODEBOOK convention): OLS of earnings_m12 on treatment + site indicators,
HC2 standard errors, two-sided t-test of the treatment coefficient.

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf

OUTCOME = "earnings_m12"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    d = pd.read_csv(a.data)
    r = smf.ols(f"{OUTCOME} ~ treatment + C(site)", d).fit(cov_type="HC2", use_t=True)
    out = {
        "estimate": float(r.params["treatment"]),
        "std_error": float(r.bse["treatment"]),
        "p_value": float(r.pvalues["treatment"]),
    }
    with open(a.output, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
