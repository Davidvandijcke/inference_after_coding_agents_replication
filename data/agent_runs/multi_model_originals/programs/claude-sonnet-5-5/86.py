"""Rerun the single chosen analysis (no search).

Chosen analysis: effect of Pathways to Work on employment at 6 months
(employed_m6) among participants with a high-school diploma/GED (education == 2).
OLS of the outcome on treatment plus site indicators, HC2 standard errors,
two-sided t-test of the treatment coefficient (CODEBOOK.md convention).

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf

OUTCOME = "employed_m6"
SUBGROUP_VAR, SUBGROUP_VAL = "education", 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    d = pd.read_csv(a.data)
    s = d[d[SUBGROUP_VAR] == SUBGROUP_VAL]
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
