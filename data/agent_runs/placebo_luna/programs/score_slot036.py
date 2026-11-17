#!/usr/bin/env python3
"""Estimate the block-adjusted year-2 sick-day treatment effect."""

import argparse
import json

import pandas as pd
import statsmodels.api as sm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="CSV file with evaluation data")
    parser.add_argument("--output", required=True, help="Path for JSON result")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    columns = ["any_sick_days_y2", "any_sick_days_base", "treatment", "block"]
    sample = data[columns].dropna()
    block_terms = pd.get_dummies(sample["block"], prefix="block", drop_first=True, dtype=float)
    design = pd.concat(
        [sample[["treatment", "any_sick_days_base"]], block_terms], axis=1
    )
    design = sm.add_constant(design)
    fit = sm.OLS(sample["any_sick_days_y2"], design).fit(cov_type="HC2")

    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    with open(args.output, "w", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
