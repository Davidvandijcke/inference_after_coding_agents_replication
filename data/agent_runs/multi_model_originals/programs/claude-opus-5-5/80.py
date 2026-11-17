"""Rerun the chosen Pathways to Work analysis on a dataset with the data.csv columns.

Test: effect of program assignment on a material-hardship count among parents.
  Sample:  parent == 1
  Outcome: food_insecure_m6 + debt_collections_m6 + food_insecure_m12 + debt_collections_m12  (0-4)
  Model:   OLS of outcome on treatment + site indicators (no other covariates)
  SEs:     HC2; two-sided t-test of the treatment coefficient (t reference with
           residual degrees of freedom)

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf

HARDSHIP_ITEMS = [
    "food_insecure_m6",
    "debt_collections_m6",
    "food_insecure_m12",
    "debt_collections_m12",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True, help="CSV with the data.csv columns")
    parser.add_argument("--output", required=True, help="Path for the JSON result")
    args = parser.parse_args()

    d = pd.read_csv(args.data)
    parents = d[d["parent"] == 1].copy()
    parents["hardship"] = parents[HARDSHIP_ITEMS].sum(axis=1)

    fit = smf.ols("hardship ~ treatment + C(site)", data=parents).fit(cov_type="HC2", use_t=True)
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
