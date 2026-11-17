"""Rerun the single chosen analysis: effect of Pathways to Work on 12-month food
insecurity among male (female == 0) participants.

Convention (CODEBOOK.md): OLS of the outcome on treatment + site indicators,
HC2 standard errors, two-sided t-test of the treatment coefficient, fit within
the subgroup. No search over outcomes or subgroups is performed.

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf

OUTCOME = "food_insecure_m12"
SUBGROUP = "female == 0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    d = pd.read_csv(args.data).query(SUBGROUP)
    fit = smf.ols(f"{OUTCOME} ~ treatment + C(site)", d).fit(cov_type="HC2", use_t=True)
    out = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
