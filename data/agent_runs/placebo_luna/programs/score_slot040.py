#!/usr/bin/env python3
"""Score the block-adjusted treatment effect on partners' desired employment."""

import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf


OUTCOME = "partner_employment_1y_fu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="CSV dataset")
    parser.add_argument("--output", required=True, help="Path for JSON results")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = {"treatment", "block", OUTCOME}
    missing = required.difference(data.columns)
    if missing:
        parser.error(f"missing required columns: {', '.join(sorted(missing))}")

    analysis = data[["treatment", "block", OUTCOME]].dropna()
    model = smf.ols(
        f'Q("{OUTCOME}") ~ treatment + C(block)', data=analysis
    ).fit(cov_type="HC2", use_t=True)

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
