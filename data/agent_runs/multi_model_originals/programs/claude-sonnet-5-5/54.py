#!/usr/bin/env python
"""Re-run the single chosen Pathways to Work analysis on another dataset.

Chosen analysis: effect of the program on self-rated health at 6 months
(health_m6) among participants with some college or more (education >= 3).
Convention from CODEBOOK.md: OLS of the outcome on treatment plus site
indicators, HC2 robust standard errors, two-sided t-test on treatment.

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf

OUTCOME = "health_m6"
SUBGROUP_MIN_EDUCATION = 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    d = pd.read_csv(a.data)
    d = d[d["education"] >= SUBGROUP_MIN_EDUCATION]
    m = smf.ols(f"{OUTCOME} ~ treatment + C(site)", data=d).fit(cov_type="HC2")
    out = {
        "estimate": float(m.params["treatment"]),
        "std_error": float(m.bse["treatment"]),
        "p_value": float(m.pvalues["treatment"]),
    }
    with open(a.output, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
