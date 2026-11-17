#!/usr/bin/env python3
"""Score the randomized treatment effect on partner employment plans in age group 2."""

import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Input CSV dataset")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    sample = data.loc[data["age_group"] == 2,
                      ["treatment", "block", "partner_employment_1y_fu"]].dropna()
    model = smf.ols(
        'partner_employment_1y_fu ~ treatment + C(block)', data=sample
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
