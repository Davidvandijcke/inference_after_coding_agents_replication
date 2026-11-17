#!/usr/bin/env python
"""Re-run the single chosen analysis: effect of Pathways to Work on food insecurity
at 6 months (food_insecure_m6), full sample.

Model (CODEBOOK convention): OLS of food_insecure_m6 on treatment + site indicators,
HC2 robust standard errors, two-sided t-test of the treatment coefficient.

Usage: python score.py --data <path> --output <path>
"""
import argparse, json
import pandas as pd
import statsmodels.formula.api as smf

OUTCOME = "food_insecure_m6"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    d = pd.read_csv(a.data)
    m = smf.ols(f"{OUTCOME} ~ treatment + C(site)", d).fit(cov_type="HC2", use_t=True)
    res = {
        "estimate": float(m.params["treatment"]),
        "std_error": float(m.bse["treatment"]),
        "p_value": float(m.pvalues["treatment"]),
    }
    with open(a.output, "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
