"""Pathways to Work: effect of program assignment on employment at 12 months.

Single prespecified-style analysis (no search): OLS of employed_m12 on treatment,
site indicators, and baseline covariates, with HC2 robust standard errors and a
two-sided t-test of the treatment coefficient (CODEBOOK.md analysis convention).

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf

OUTCOME = "employed_m12"
COVARIATES = [
    "age", "female", "parent", "education", "household_size",
    "employed_baseline", "earnings_baseline", "months_unemployed_baseline",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    d = pd.read_csv(args.data)
    d = d.dropna(subset=[OUTCOME, "treatment", "site"] + COVARIATES)
    rhs = ["treatment"]
    if d["site"].nunique() > 1:
        rhs.append("C(site)")
    # drop constant covariates so the design stays full rank
    rhs += [c for c in COVARIATES if d[c].nunique() > 1]
    fit = smf.ols(f"{OUTCOME} ~ " + " + ".join(rhs), data=d).fit(cov_type="HC2", use_t=True)

    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
