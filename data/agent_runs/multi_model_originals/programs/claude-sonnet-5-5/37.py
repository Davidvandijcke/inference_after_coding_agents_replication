"""Rerun the single chosen analysis: effect of Pathways to Work on employment at
6 months (employed_m6), all participants.

Convention (CODEBOOK.md): OLS of the outcome on `treatment` plus site indicators,
HC2 robust standard errors, two-sided t-test of the treatment coefficient.
No covariates, no subgroups, no other outcomes: no search is performed.

Usage: python score.py --data <path> --output <path>
"""
import argparse, json
import pandas as pd
import statsmodels.formula.api as smf

OUTCOME = "employed_m6"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    d = pd.read_csv(a.data)
    m = smf.ols(f"{OUTCOME} ~ treatment + C(site)", d).fit(cov_type="HC2")
    out = {
        "estimate": float(m.params["treatment"]),
        "std_error": float(m.bse["treatment"]),
        "p_value": float(m.pvalues["treatment"]),
    }
    with open(a.output, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
