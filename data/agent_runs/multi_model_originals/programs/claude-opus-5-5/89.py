"""Rerun the chosen Pathways to Work analysis on a dataset with the codebook's columns.

Chosen test: effect of assignment to the program on employment at 12 months
(employed_m12) among participants who were long-term unemployed at random
assignment (months_unemployed_baseline >= 6). Following the CODEBOOK analysis
convention, within that subgroup: OLS of employed_m12 on treatment plus site
indicators, HC2 robust standard error, two-sided t-test of the treatment
coefficient (residual degrees of freedom).

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf

OUTCOME = "employed_m12"
LTU_CUTOFF_MONTHS = 6


def score(df):
    sub = df[df["months_unemployed_baseline"] >= LTU_CUTOFF_MONTHS]
    if sub.empty or sub["treatment"].nunique() < 2:
        raise ValueError("Subgroup needs both treatment and control participants.")
    formula = f"{OUTCOME} ~ treatment"
    if sub["site"].nunique() > 1:
        formula += " + C(site)"
    fit = smf.ols(formula, data=sub).fit(cov_type="HC2", use_t=True)
    return {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = score(pd.read_csv(args.data))
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
