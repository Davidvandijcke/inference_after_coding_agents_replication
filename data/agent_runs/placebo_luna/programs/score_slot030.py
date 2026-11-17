#!/usr/bin/env python3
"""Score the year-two randomized effect on physical problems."""

import argparse
import json

import pandas as pd
import statsmodels.api as sm
from scipy.stats import t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="CSV dataset")
    parser.add_argument("--output", required=True, help="Path for JSON results")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    columns = ["physical_problems_y2", "physical_problems_base", "treatment", "block"]
    sample = data[columns].dropna()

    # OLS with treatment, block indicators, and the matching baseline outcome.
    block_indicators = pd.get_dummies(
        sample["block"].astype("category"), prefix="block", drop_first=True, dtype=float
    )
    predictors = pd.concat(
        [
            sample[["treatment", "physical_problems_base"]].astype(float),
            block_indicators,
        ],
        axis=1,
    )
    model = sm.OLS(
        sample["physical_problems_y2"].astype(float), sm.add_constant(predictors)
    ).fit(cov_type="HC2", use_t=True)

    estimate = float(model.params["treatment"])
    std_error = float(model.bse["treatment"])
    p_value = float(2 * t.sf(abs(estimate / std_error), model.df_resid))
    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(
            {"estimate": estimate, "std_error": std_error, "p_value": p_value},
            output_file,
            indent=2,
        )
        output_file.write("\n")


if __name__ == "__main__":
    main()
