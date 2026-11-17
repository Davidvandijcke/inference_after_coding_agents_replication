#!/usr/bin/env python3
"""Score the year 1 intention-to-treat effect on current smoking."""

import argparse
import json

import pandas as pd
import statsmodels.api as sm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Input CSV file")
    parser.add_argument("--output", required=True, help="Output JSON file")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = ["current_smoker_y1", "current_smoker_base", "treatment", "block"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        parser.error("missing required columns: " + ", ".join(missing))

    sample = data.dropna(subset=required).copy()
    block_indicators = pd.get_dummies(
        sample["block"].astype("category"),
        prefix="block",
        drop_first=True,
        dtype=float,
    )
    design = pd.concat(
        [
            pd.Series(1.0, index=sample.index, name="intercept"),
            sample["treatment"].astype(float).rename("treatment"),
            block_indicators,
            sample["current_smoker_base"].astype(float).rename("baseline"),
        ],
        axis=1,
    )
    result = sm.OLS(sample["current_smoker_y1"].astype(float), design).fit(cov_type="HC2")

    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(
            {
                "estimate": float(result.params["treatment"]),
                "std_error": float(result.bse["treatment"]),
                "p_value": float(result.pvalues["treatment"]),
            },
            output_file,
            indent=2,
        )
        output_file.write("\n")


if __name__ == "__main__":
    main()
