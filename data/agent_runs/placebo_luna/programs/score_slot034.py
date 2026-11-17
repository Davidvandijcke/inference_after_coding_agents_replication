#!/usr/bin/env python3
"""Score the prespecified lead analysis on a compatible evaluation dataset."""

import argparse
import json

import pandas as pd
import statsmodels.api as sm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    # Age group 3 is also randomization block 3 (participants aged 36–40).
    sample = data.loc[
        data["age_group"].eq(3),
        ["plan_employment_1y_post", "treatment", "block"],
    ].dropna()
    if sample["treatment"].nunique() != 2:
        raise ValueError("The analysis sample must contain both treatment arms")

    y = sample["plan_employment_1y_post"].astype(float)
    design = pd.DataFrame({"treatment": sample["treatment"].astype(float)})
    block_dummies = pd.get_dummies(sample["block"], drop_first=True, dtype=float)
    design = pd.concat([design.reset_index(drop=True), block_dummies.reset_index(drop=True)], axis=1)
    design = sm.add_constant(design, has_constant="add")
    result = sm.OLS(y.reset_index(drop=True), design).fit(cov_type="HC2", use_t=True)

    with open(args.output, "w", encoding="utf-8") as output:
        json.dump(
            {
                "estimate": float(result.params["treatment"]),
                "std_error": float(result.bse["treatment"]),
                "p_value": float(result.pvalues["treatment"]),
            },
            output,
            indent=2,
        )
        output.write("\n")


if __name__ == "__main__":
    main()
