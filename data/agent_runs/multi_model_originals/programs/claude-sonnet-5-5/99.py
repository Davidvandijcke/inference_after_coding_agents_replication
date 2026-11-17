"""Rerun the chosen analysis: effect of treatment on 12-month employment.

OLS of employed_m12 on treatment and site indicators, HC2 robust standard
errors, two-sided t-test on the treatment coefficient (CODEBOOK.md convention).
No search or subgroup selection is performed.

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf

OUTCOME = "employed_m12"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    df = pd.read_csv(a.data)
    m = smf.ols(f"{OUTCOME} ~ treatment + C(site)", df).fit(cov_type="HC2")
    out = {
        "estimate": float(m.params["treatment"]),
        "std_error": float(m.bse["treatment"]),
        "p_value": float(m.pvalues["treatment"]),
    }
    with open(a.output, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
