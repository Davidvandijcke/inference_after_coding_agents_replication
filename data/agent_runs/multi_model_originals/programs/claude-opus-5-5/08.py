"""Rerun the chosen Pathways to Work analysis on a dataset with the same columns.

Chosen test (selected after exploratory search; see report): effect of program
assignment on enrollment in education/job training at 12 months
(`training_m12`) among mothers (female == 1 and parent == 1).

Convention (CODEBOOK.md): OLS of the outcome on `treatment` and site indicators,
HC2 heteroskedasticity-robust standard errors, two-sided t-test of the
`treatment` coefficient (residual degrees of freedom).

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf

OUTCOME = "training_m12"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True, help="CSV with the codebook columns")
    parser.add_argument("--output", required=True, help="path for the JSON result")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    mothers = df[(df["female"] == 1) & (df["parent"] == 1)]

    site_term = " + C(site)" if mothers["site"].nunique() > 1 else ""
    fit = smf.ols(f"{OUTCOME} ~ treatment{site_term}", data=mothers).fit(
        cov_type="HC2", use_t=True
    )

    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
