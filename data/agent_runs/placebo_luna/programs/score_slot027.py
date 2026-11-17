#!/usr/bin/env python3
"""Score the prespecified year-two sick-day analysis on a supplied CSV."""
import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf


def main():
    parser = argparse.ArgumentParser(
        description="Estimate the block- and baseline-adjusted treatment effect on any_sick_days_y2."
    )
    parser.add_argument("--data", required=True, help="Input CSV with evaluation columns")
    parser.add_argument("--output", required=True, help="Path for JSON result")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = ["any_sick_days_y2", "treatment", "block", "any_sick_days_base"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        parser.error("input is missing required columns: " + ", ".join(missing))

    analysis = data[required].dropna()
    if analysis.empty or analysis["treatment"].nunique() != 2:
        parser.error("complete cases must include observations in both treatment arms")

    model = smf.ols(
        "any_sick_days_y2 ~ treatment + C(block) + any_sick_days_base",
        data=analysis,
    ).fit(cov_type="HC2", use_t=True)

    result = {
        "estimate": float(model.params["treatment"]),
        "std_error": float(model.bse["treatment"]),
        "p_value": float(model.pvalues["treatment"]),
    }
    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2)
        output_file.write("\n")


if __name__ == "__main__":
    main()
