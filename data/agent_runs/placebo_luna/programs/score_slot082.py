#!/usr/bin/env python3
"""Estimate the block-adjusted treatment effect on discussing the video with friends."""

import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="CSV dataset with the evaluation columns")
    parser.add_argument("--output", required=True, help="Path for the JSON result")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    analysis = data[["discussed_friends_fu", "treatment", "block"]].dropna()
    fit = smf.ols("discussed_friends_fu ~ treatment + C(block)", data=analysis).fit(
        cov_type="HC2"
    )
    estimate = float(fit.params["treatment"])
    std_error = float(fit.bse["treatment"])
    statistic = estimate / std_error
    p_value = float(2 * t.sf(abs(statistic), df=int(fit.df_resid)))

    with open(args.output, "w", encoding="utf-8") as output:
        json.dump(
            {"estimate": estimate, "std_error": std_error, "p_value": p_value},
            output,
            indent=2,
        )
        output.write("\n")


if __name__ == "__main__":
    main()
