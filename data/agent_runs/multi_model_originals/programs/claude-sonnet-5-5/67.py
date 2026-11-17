#!/usr/bin/env python
"""Re-run the single chosen analysis from the Pathways to Work exploration.

Chosen analysis (exploratory, selected post hoc): effect of assignment to the
program on food insecurity at 6 months (food_insecure_m6) among participants
with less than a high school education (education == 1).

Estimator follows the CODEBOOK convention: OLS of the outcome on `treatment`
plus site indicators, HC2 standard errors, two-sided t-test of the treatment
coefficient. No searching over outcomes, subgroups or follow-up points is done.

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import numpy as np
import pandas as pd
import statsmodels.api as sm

OUTCOME = "food_insecure_m6"
SUBGROUP_VAR, SUBGROUP_VALUE = "education", 1


def run(df):
    sub = df[df[SUBGROUP_VAR] == SUBGROUP_VALUE]
    sites = pd.get_dummies(sub["site"], prefix="site", dtype=float)
    X = pd.concat([sub[["treatment"]].astype(float), sites], axis=1)
    fit = sm.OLS(sub[OUTCOME].astype(float), X).fit(cov_type="HC2")
    # t-test (not normal) on residual df, as in statsmodels' use_t
    fit_t = sm.OLS(sub[OUTCOME].astype(float), X).fit(cov_type="HC2", use_t=True)
    return {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit_t.pvalues["treatment"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    res = run(pd.read_csv(args.data))
    with open(args.output, "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
