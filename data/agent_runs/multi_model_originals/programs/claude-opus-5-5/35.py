"""Pathways to Work: covariate-adjusted effect of program assignment on
employment at 12 months (full sample).

OLS of employed_m12 on treatment, site indicators and fixed baseline
covariates; HC2 robust standard errors; two-sided t-test on treatment.

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf

FORMULA = (
    "employed_m12 ~ treatment + C(site)"
    " + age + female + parent + C(education) + household_size"
    " + employed_baseline + earnings_baseline + months_unemployed_baseline"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    d = pd.read_csv(args.data)
    m = smf.ols(FORMULA, data=d).fit(cov_type="HC2", use_t=True)
    out = {
        "estimate": float(m.params["treatment"]),
        "std_error": float(m.bse["treatment"]),
        "p_value": float(m.pvalues["treatment"]),
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
