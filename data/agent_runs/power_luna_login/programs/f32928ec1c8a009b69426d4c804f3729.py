#!/usr/bin/env python3
"""Run the selected Pathways to Work analysis on a supplied CSV dataset."""

import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to the participant CSV")
    parser.add_argument("--output", required=True, help="Path for the JSON result")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = {"parent", "employed_m12", "treatment", "site"}
    missing = required.difference(data.columns)
    if missing:
        parser.error("missing required columns: " + ", ".join(sorted(missing)))

    parents = data.loc[data["parent"] == 1]
    model = smf.ols("employed_m12 ~ treatment + C(site)", data=parents).fit(
        cov_type="HC2"
    )
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
