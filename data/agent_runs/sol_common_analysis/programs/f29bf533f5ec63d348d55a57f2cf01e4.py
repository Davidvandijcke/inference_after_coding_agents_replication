#!/usr/bin/env python3
"""Estimate the assignment effect on employment at 12 months.

Usage: python score.py --data data.csv --output result.json
Dependencies: pandas, statsmodels (and their dependencies).
"""

import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf


FORMULA = (
    "employed_m12 ~ treatment + C(site) + age + female + parent"
    " + C(education) + household_size + employed_baseline"
    " + earnings_baseline + months_unemployed_baseline"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input participant CSV")
    parser.add_argument("--output", required=True, help="Output JSON file")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    model = smf.ols(FORMULA, data=data, missing="raise").fit(
        cov_type="HC2", use_t=True
    )
    result = {
        "estimate": float(model.params["treatment"]),
        "std_error": float(model.bse["treatment"]),
        "p_value": float(model.pvalues["treatment"]),
    }
    with open(args.output, "w", encoding="utf-8") as output:
        json.dump(result, output, allow_nan=False, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
