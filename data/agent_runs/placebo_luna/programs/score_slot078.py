#!/usr/bin/env python3
"""Score the prespecified lead analysis on a dataset with the codebook columns."""

import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Input CSV")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    outcome = "other_tobacco_y1"
    baseline = "other_tobacco_base"
    required = [outcome, baseline, "treatment", "block"]
    data = pd.read_csv(args.data)
    missing = [column for column in required if column not in data.columns]
    if missing:
        parser.error("Missing required columns: " + ", ".join(missing))

    sample = data[required].dropna()
    model = smf.ols(
        f"{outcome} ~ treatment + C(block) + {baseline}", data=sample
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
