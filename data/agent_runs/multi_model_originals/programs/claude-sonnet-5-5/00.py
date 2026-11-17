"""Rerun the single chosen analysis: effect of Pathways to Work on 12-month self-rated health.

OLS of health_m12 on treatment and site indicators, HC2 robust standard errors,
two-sided t-test of the treatment coefficient (CODEBOOK.md convention).
No search over outcomes, follow-ups or subgroups is performed here.

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

OUTCOME = "health_m12"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    df = pd.read_csv(a.data)
    fit = smf.ols(f"{OUTCOME} ~ treatment + C(site)", df).fit(cov_type="HC2")
    res = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    with open(a.output, "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
