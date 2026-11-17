#!/usr/bin/env python3
"""Score the exploratory mid-age presenteeism analysis on a compatible dataset."""

import argparse
import json

import pandas as pd
import statsmodels.api as sm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    outcome = "presenteeism_score_y1"
    required = ["age_group", "treatment", "block", outcome]
    missing = [column for column in required if column not in data.columns]
    if missing:
        parser.error("missing required columns: " + ", ".join(missing))

    # This is the prespecified score for the selected subgroup analysis:
    # age group 2, with block fixed effects and HC2 robust t inference.
    sample = data.loc[data["age_group"] == 2, required].dropna()
    if sample.empty or sample["treatment"].nunique() != 2:
        parser.error("age_group 2 must have nonmissing outcomes in both treatment arms")

    block_indicators = pd.get_dummies(
        sample["block"], prefix="block", drop_first=True, dtype=float
    )
    design = pd.concat([sample[["treatment"]], block_indicators], axis=1)
    design = sm.add_constant(design)
    result = sm.OLS(sample[outcome], design).fit(cov_type="HC2", use_t=True)

    payload = {
        "estimate": float(result.params["treatment"]),
        "std_error": float(result.bse["treatment"]),
        "p_value": float(result.pvalues["treatment"]),
    }
    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2)
        output_file.write("\n")


if __name__ == "__main__":
    main()
