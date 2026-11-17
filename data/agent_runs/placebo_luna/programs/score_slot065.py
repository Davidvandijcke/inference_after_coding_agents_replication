#!/usr/bin/env python3
"""Score the exploratory male/female treatment-effect interaction."""

import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to a CSV with the evaluation columns")
    parser.add_argument("--output", required=True, help="Path for the JSON result")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    # Gender 1 and 2 are the two sufficiently represented groups in this
    # selected comparison; gender 3 is excluded from this interaction test.
    data = data.loc[data["gender"].isin([1, 2])].dropna(
        subset=["effect_below_one_to_one", "treatment", "block", "gender"]
    ).copy()
    data["female"] = (data["gender"] == 2).astype(int)

    model = smf.ols(
        "effect_below_one_to_one ~ treatment * female + C(block)", data=data
    ).fit(cov_type="HC2")
    term = "treatment:female"
    estimate = float(model.params[term])
    std_error = float(model.bse[term])
    degrees_freedom = float(model.df_resid)
    statistic = estimate / std_error
    p_value = float(2 * t.sf(abs(statistic), degrees_freedom))

    with open(args.output, "w", encoding="utf-8") as output:
        json.dump(
            {"estimate": estimate, "std_error": std_error, "p_value": p_value},
            output,
            indent=2,
        )
        output.write("\n")


if __name__ == "__main__":
    main()
