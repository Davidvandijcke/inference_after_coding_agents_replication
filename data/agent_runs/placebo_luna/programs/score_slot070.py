#!/usr/bin/env python3
"""Estimate the treatment effect on requesting premium information."""

import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import t


OUTCOME = "wants_premium_info_post"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="CSV dataset")
    parser.add_argument("--output", required=True, help="Path for JSON results")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    analysis = data[["treatment", "block", OUTCOME]].dropna()
    if analysis.empty:
        raise ValueError(f"No observed values for {OUTCOME}")

    model = smf.ols(f'Q("{OUTCOME}") ~ treatment + C(block)', data=analysis).fit(
        cov_type="HC2"
    )
    estimate = float(model.params["treatment"])
    std_error = float(model.bse["treatment"])
    statistic = estimate / std_error
    p_value = float(2 * t.sf(abs(statistic), df=model.df_resid))

    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(
            {
                "estimate": estimate,
                "std_error": std_error,
                "p_value": p_value,
            },
            output_file,
            indent=2,
        )
        output_file.write("\n")


if __name__ == "__main__":
    main()
