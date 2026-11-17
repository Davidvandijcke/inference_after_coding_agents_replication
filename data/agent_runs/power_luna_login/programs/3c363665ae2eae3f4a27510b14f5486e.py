#!/usr/bin/env python3
"""Score the 12-month employment effect in a Pathways-style evaluation."""

import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf


def main():
    parser = argparse.ArgumentParser(
        description="Estimate the site-adjusted treatment effect on 12-month employment."
    )
    parser.add_argument("--data", required=True, help="Input CSV file")
    parser.add_argument("--output", required=True, help="Output JSON file")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    model = smf.ols("employed_m12 ~ treatment + C(site)", data=data).fit(
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
