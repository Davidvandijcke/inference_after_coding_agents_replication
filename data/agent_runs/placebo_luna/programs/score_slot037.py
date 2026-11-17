#!/usr/bin/env python3
"""Score the prespecified-for-scoring exploratory finding on a new dataset.

Chosen analysis: among participants in age_group 5 (ages 46–50), regress
took_action_fu on treatment and randomization block indicators, using HC2
robust standard errors and a two-sided t test of the treatment coefficient.
"""

import argparse
import json
import sys

import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = {"age_group", "took_action_fu", "treatment", "block"}
    missing = required.difference(data.columns)
    if missing:
        parser.error("missing required columns: " + ", ".join(sorted(missing)))

    subset = data.loc[data["age_group"] == 5,
                      ["took_action_fu", "treatment", "block"]].dropna()
    if subset.empty or subset["treatment"].nunique() != 2:
        parser.error("age_group 5 must contain observed outcomes in both treatment arms")

    model = smf.ols("took_action_fu ~ treatment + C(block)", data=subset).fit(
        cov_type="HC2"
    )
    estimate = float(model.params["treatment"])
    std_error = float(model.bse["treatment"])
    p_value = float(2 * t.sf(abs(estimate / std_error), model.df_resid))

    with open(args.output, "w", encoding="utf-8") as output:
        json.dump(
            {"estimate": estimate, "std_error": std_error, "p_value": p_value},
            output,
            indent=2,
        )
        output.write("\n")


if __name__ == "__main__":
    main()
