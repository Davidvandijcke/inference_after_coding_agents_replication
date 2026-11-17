#!/usr/bin/env python3
"""Estimate the block-adjusted program effect on pension-overview interest."""

import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf


OUTCOME = "wants_pension_overview_post"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="CSV file with the evaluation columns")
    parser.add_argument("--output", required=True, help="Path for the JSON result")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    analysis = data[["treatment", "block", OUTCOME]].dropna()
    if analysis.empty or analysis["treatment"].nunique() != 2:
        raise ValueError("Outcome data must include observations in both treatment arms")

    fit = smf.ols(f'{OUTCOME} ~ treatment + C(block)', data=analysis).fit(cov_type="HC2")
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
