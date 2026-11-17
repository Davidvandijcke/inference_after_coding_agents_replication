#!/usr/bin/env python3
"""Score the year 1 sick-days result among nonwhite participants.

Usage: python score.py --data <path> --output <path>

Fits OLS of any_sick_days_y1 on treatment, randomization-block indicators,
and baseline any_sick_days_base, using HC2 robust standard errors and a
two-sided t test of the treatment coefficient.
"""

import argparse
import json

import pandas as pd
import statsmodels.api as sm


def analyze(data_path):
    data = pd.read_csv(data_path)
    required = {
        "white", "treatment", "block", "any_sick_days_y1",
        "any_sick_days_base",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    sample = data.loc[data["white"] == 0].dropna(
        subset=["treatment", "block", "any_sick_days_y1", "any_sick_days_base"]
    )
    if sample.empty or sample["treatment"].nunique() != 2:
        raise ValueError("The nonwhite complete-case sample must contain both arms")

    predictors = sample[["treatment", "block", "any_sick_days_base"]].copy()
    predictors = pd.get_dummies(
        predictors, columns=["block"], drop_first=True, dtype=float
    )
    predictors = sm.add_constant(predictors.astype(float), has_constant="add")
    fit = sm.OLS(sample["any_sick_days_y1"].astype(float), predictors).fit(
        cov_type="HC2", use_t=True
    )

    return {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()
    result = analyze(args.data)
    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2)
        output_file.write("\n")


if __name__ == "__main__":
    main()
