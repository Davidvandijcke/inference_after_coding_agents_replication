#!/usr/bin/env python3
"""Score the exploratory treatment effect among WTP-good nonusers."""

import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="CSV file with evaluation data")
    parser.add_argument("--output", required=True, help="Path for JSON result")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = {
        "block",
        "treatment",
        "uses_wtp_good",
        "effect_below_one_to_one",
    }
    missing = required.difference(data.columns)
    if missing:
        parser.error("data is missing required columns: " + ", ".join(sorted(missing)))

    sample = data.loc[
        data["uses_wtp_good"].eq(0),
        ["block", "treatment", "effect_below_one_to_one"],
    ].dropna()
    if sample.empty:
        parser.error("no complete observations among uses_wtp_good == 0")

    result = smf.ols(
        "effect_below_one_to_one ~ treatment + C(block)", data=sample
    ).fit(cov_type="HC2", use_t=True)
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
