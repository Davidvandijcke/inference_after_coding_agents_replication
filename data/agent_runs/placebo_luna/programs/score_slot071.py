#!/usr/bin/env python3
"""Score the treatment effect on perceived consumption impact in block 5.

Runs OLS of effect_below_one_to_one on treatment among respondents assigned
to block 5 (green electricity), using HC2 robust standard errors.
"""

import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = {"block", "treatment", "effect_below_one_to_one"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    subset = data.loc[
        data["block"] == 5,
        ["treatment", "effect_below_one_to_one"],
    ].dropna()
    if subset.empty or subset["treatment"].nunique() != 2:
        raise ValueError("Block 5 must contain observed outcomes in both treatment arms")

    result = smf.ols("effect_below_one_to_one ~ treatment", data=subset).fit(
        cov_type="HC2"
    )
    payload = {
        "estimate": float(result.params["treatment"]),
        "std_error": float(result.bse["treatment"]),
        "p_value": float(result.pvalues["treatment"]),
    }
    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, allow_nan=False)
        output_file.write("\n")


if __name__ == "__main__":
    main()
