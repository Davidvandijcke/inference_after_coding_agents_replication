#!/usr/bin/env python
"""Rerun the single chosen analysis from the Pathways to Work exploration.

Chosen (exploratory) analysis: effect of assignment to the program on
`debt_collections_m6` among participants with education == 4 (bachelor's or
higher). OLS of the outcome on `treatment` plus site indicators, HC2 robust
standard errors, two-sided t-test of the treatment coefficient (CODEBOOK.md
convention). No search is performed here.

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf

OUTCOME = "debt_collections_m6"
SUBGROUP_COL, SUBGROUP_VAL = "education", 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    d = pd.read_csv(a.data)
    s = d[d[SUBGROUP_COL] == SUBGROUP_VAL]
    fit = smf.ols(f"{OUTCOME} ~ treatment + C(site)", s).fit(cov_type="HC2", use_t=True)
    out = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    with open(a.output, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
