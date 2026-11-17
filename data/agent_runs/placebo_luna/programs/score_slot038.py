#!/usr/bin/env python3
"""Score the selected subgroup treatment effect in an evaluation dataset."""

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
    subgroup = data.loc[data["uses_topic_good"] == 0].dropna(
        subset=["wtp_zero_if_ineffective", "treatment", "block"]
    )
    if subgroup.empty:
        raise ValueError("No complete observations in uses_topic_good == 0 subgroup")

    model = smf.ols(
        "wtp_zero_if_ineffective ~ treatment + C(block)", data=subgroup
    ).fit(cov_type="HC2")
    estimate = float(model.params["treatment"])
    std_error = float(model.bse["treatment"])
    statistic = estimate / std_error
    p_value = float(2 * t.sf(abs(statistic), df=float(model.df_resid)))

    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(
            {"estimate": estimate, "std_error": std_error, "p_value": p_value},
            output_file,
            indent=2,
        )
        output_file.write("\n")


if __name__ == "__main__":
    main()
