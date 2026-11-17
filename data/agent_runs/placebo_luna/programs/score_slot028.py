#!/usr/bin/env python3
"""Score the randomized treatment effect on immediate hopefulness."""

import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    sample = data[["felt_hopeful_post", "treatment", "block"]].dropna()
    fit = smf.ols(
        'Q("felt_hopeful_post") ~ treatment + C(block)', data=sample
    ).fit(cov_type="HC2")

    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    with open(args.output, "w", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
