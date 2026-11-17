#!/usr/bin/env python3
"""Score the block-adjusted treatment effect on desired 10-year employment post."""

import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf


OUTCOME = "plan_employment_10y_post"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Input CSV dataset")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    analysis = data[[OUTCOME, "treatment", "block"]].dropna()
    if analysis.empty or analysis["treatment"].nunique() != 2:
        raise ValueError("The analysis requires observed outcomes in both treatment arms.")

    result = smf.ols(f'{OUTCOME} ~ treatment + C(block)', data=analysis).fit(
        cov_type="HC2", use_t=True
    )
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
