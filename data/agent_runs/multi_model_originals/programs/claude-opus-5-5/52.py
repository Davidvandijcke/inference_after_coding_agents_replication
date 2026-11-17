"""Pathways to Work: effect of program assignment on psychological distress
(Kessler-6) at 12 months, full sample.

Pre-chosen specification (no search): OLS of distress_m12 on treatment and
site indicators, HC2 robust standard errors, two-sided t-test on treatment.

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

    df = pd.read_csv(args.data)
    fit = smf.ols("distress_m12 ~ treatment + C(site)", data=df).fit(cov_type="HC2", use_t=True)
    out = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
