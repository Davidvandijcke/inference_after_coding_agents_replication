#!/usr/bin/env python
"""Pathways to Work: rerun the single chosen analysis.

Effect of random assignment to the program on employment at 12 months
(employed_m12). OLS of employed_m12 on treatment, site indicators and the
pre-specified baseline covariate set; HC2 robust standard errors; two-sided
t-test of the treatment coefficient (residual degrees of freedom).

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

OUTCOME = "employed_m12"
FORMULA = (
    f"{OUTCOME} ~ treatment + C(site) + age + female + parent + C(education)"
    " + household_size + employed_baseline + earnings_baseline"
    " + months_unemployed_baseline"
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    fit = smf.ols(FORMULA, data=df).fit(cov_type="HC2", use_t=True)
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
