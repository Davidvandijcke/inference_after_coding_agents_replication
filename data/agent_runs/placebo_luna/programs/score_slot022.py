#!/usr/bin/env python3
"""Score the block-adjusted treatment effect on next-year plans at follow-up."""

import argparse
import json

import pandas as pd
from scipy.stats import t
import statsmodels.formula.api as smf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="CSV dataset")
    parser.add_argument("--output", required=True, help="Path for JSON results")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    outcome = "plan_employment_1y_fu"
    sample = data[["treatment", "block", outcome]].dropna()
    fit = smf.ols(f'Q("{outcome}") ~ treatment + C(block)', data=sample).fit(
        cov_type="HC2"
    )
    estimate = float(fit.params["treatment"])
    std_error = float(fit.bse["treatment"])
    statistic = estimate / std_error
    p_value = float(2 * t.sf(abs(statistic), df=fit.df_resid))

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
