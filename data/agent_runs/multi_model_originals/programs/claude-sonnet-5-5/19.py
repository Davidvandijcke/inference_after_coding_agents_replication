#!/usr/bin/env python
"""Rerun the single chosen analysis (no search).

Chosen analysis: effect of Pathways to Work on food insecurity at 12 months
(food_insecure_m12) among participants aged 30-44 (30 <= age < 45).
Convention from CODEBOOK.md: OLS of the outcome on treatment + site indicators,
HC2 robust standard errors, two-sided t-test on the treatment coefficient.

Usage: python score.py --data <path> --output <path>
"""
import argparse, json
import pandas as pd
import statsmodels.formula.api as smf

OUTCOME = "food_insecure_m12"
AGE_MIN, AGE_MAX = 30, 45  # 30 <= age < 45


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    df = pd.read_csv(a.data)
    sub = df[(df["age"] >= AGE_MIN) & (df["age"] < AGE_MAX)]
    m = smf.ols(f"{OUTCOME} ~ treatment + C(site)", sub).fit(cov_type="HC2", use_t=True)
    out = {
        "estimate": float(m.params["treatment"]),
        "std_error": float(m.bse["treatment"]),
        "p_value": float(m.pvalues["treatment"]),
    }
    with open(a.output, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
