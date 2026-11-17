#!/usr/bin/env python
"""Re-run the single chosen analysis: effect of Pathways to Work on
debt_collections_m6 among participants who live with a child under 18 (parent==1).

OLS of the outcome on treatment + site indicators, HC2 standard errors,
two-sided t-test of the treatment coefficient (CODEBOOK.md convention).
No search is performed.

Usage: python score.py --data <path> --output <path>
"""
import argparse, json
import pandas as pd
import statsmodels.formula.api as smf

OUTCOME = "debt_collections_m6"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    d = pd.read_csv(a.data)
    sub = d[d["parent"] == 1]
    formula = f"{OUTCOME} ~ treatment + C(site)" if sub["site"].nunique() > 1 else f"{OUTCOME} ~ treatment"
    res = smf.ols(formula, sub).fit(cov_type="HC2", use_t=True)
    out = {
        "estimate": float(res.params["treatment"]),
        "std_error": float(res.bse["treatment"]),
        "p_value": float(res.pvalues["treatment"]),
    }
    with open(a.output, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
