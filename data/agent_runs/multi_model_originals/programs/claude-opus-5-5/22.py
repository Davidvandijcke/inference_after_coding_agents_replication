"""Rerun the chosen Pathways to Work analysis on a dataset with the codebook columns.

Chosen test (exploratory finding from the original sample):
    Effect of assignment to Pathways on a 12-month summary index of eight
    outcomes, among men (female == 0).

Index: each 12-month outcome is standardized by the mean and SD of the full-sample
control group, signed so that higher = better (food insecurity, debt in
collections and distress are reversed), summed, and the sum is re-standardized
to mean 0 / SD 1 in the full-sample control group. The estimate is therefore in
control-group standard deviations.

Estimation follows CODEBOOK.md: OLS of the index on `treatment` and site
indicators within the subgroup, HC2 standard errors, two-sided t-test of the
treatment coefficient.

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf

INDEX_COMPONENTS = {
    "employed_m12": 1,
    "hours_m12": 1,
    "earnings_m12": 1,
    "fin_security_m12": 1,
    "food_insecure_m12": -1,
    "debt_collections_m12": -1,
    "distress_m12": -1,
    "health_m12": 1,
}


def build_index(df):
    control = df[df["treatment"] == 0]
    total = sum(
        sign * (df[col] - control[col].mean()) / control[col].std()
        for col, sign in INDEX_COMPONENTS.items()
    )
    ctrl_total = total[df["treatment"] == 0]
    return (total - ctrl_total.mean()) / ctrl_total.std()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    needed = ["site", "treatment", "female", *INDEX_COMPONENTS]
    df = pd.read_csv(args.data).dropna(subset=needed).copy()
    df["outcome_index"] = build_index(df)

    men = df[df["female"] == 0]
    fit = smf.ols("outcome_index ~ treatment + C(site)", data=men).fit(
        cov_type="HC2", use_t=True
    )

    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    with open(args.output, "w") as fh:
        json.dump(result, fh, indent=2)


if __name__ == "__main__":
    main()
