#!/usr/bin/env python3
"""Score the prespecified exploratory finding: year-one hospital stay."""

import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf


def main():
    parser = argparse.ArgumentParser(
        description="Estimate the block- and baseline-adjusted program effect on hospital_stay_y1."
    )
    parser.add_argument("--data", required=True, help="Input CSV with evaluation columns")
    parser.add_argument("--output", required=True, help="Path for the JSON result")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = {"hospital_stay_y1", "hospital_stay_base", "treatment", "block"}
    missing = sorted(required.difference(data.columns))
    if missing:
        parser.error("dataset is missing required columns: " + ", ".join(missing))

    analysis = data[
        ["hospital_stay_y1", "treatment", "block", "hospital_stay_base"]
    ].dropna()
    model = smf.ols(
        "hospital_stay_y1 ~ treatment + C(block) + hospital_stay_base", data=analysis
    ).fit(cov_type="HC2")

    result = {
        "estimate": float(model.params["treatment"]),
        "std_error": float(model.bse["treatment"]),
        "p_value": float(model.pvalues["treatment"]),
    }
    with open(args.output, "w", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
