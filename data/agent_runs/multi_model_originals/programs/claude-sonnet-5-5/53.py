#!/usr/bin/env python
"""Re-run the single chosen Pathways to Work analysis on another dataset.

Chosen analysis: effect of `treatment` on 12-month psychological distress
(`distress_m12`) among participants with a bachelor's degree or higher
(`education == 4`), by OLS with site indicators, HC2 robust SEs and a
two-sided t-test on the treatment coefficient (CODEBOOK.md convention).

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import numpy as np
import pandas as pd
import statsmodels.api as sm

OUTCOME = "distress_m12"
SUBGROUP_COL, SUBGROUP_VAL = "education", 4


def run(df):
    sub = df[df[SUBGROUP_COL] == SUBGROUP_VAL]
    sites = pd.get_dummies(sub["site"], prefix="site", drop_first=True, dtype=float)
    X = pd.concat([sub[["treatment"]].astype(float), sites], axis=1)
    X = sm.add_constant(X, has_constant="add")
    res = sm.OLS(sub[OUTCOME].astype(float), X).fit(cov_type="HC2", use_t=True)
    return (float(res.params["treatment"]), float(res.bse["treatment"]),
            float(res.pvalues["treatment"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    est, se, p = run(pd.read_csv(a.data))
    out = {"estimate": est, "std_error": se, "p_value": p}
    with open(a.output, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
