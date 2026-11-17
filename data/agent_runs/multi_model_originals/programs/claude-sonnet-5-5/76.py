"""Rerun the single chosen analysis (no search).

Effect of Pathways to Work on food insecurity at 6 months (food_insecure_m6)
among participants in households of 4 or more people (household_size >= 4).
OLS of outcome on treatment + site indicators, HC2 robust SEs, two-sided
t-test on the treatment coefficient (CODEBOOK.md convention).

Usage: python score.py --data <path> --output <path>
"""
import argparse, json
import pandas as pd
import statsmodels.formula.api as smf

OUTCOME = "food_insecure_m6"
MIN_HOUSEHOLD = 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    d = pd.read_csv(a.data)
    sub = d[d["household_size"] >= MIN_HOUSEHOLD]
    r = smf.ols(f"{OUTCOME} ~ treatment + C(site)", sub).fit(cov_type="HC2", use_t=True)
    out = {
        "estimate": float(r.params["treatment"]),
        "std_error": float(r.bse["treatment"]),
        "p_value": float(r.pvalues["treatment"]),
    }
    with open(a.output, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
