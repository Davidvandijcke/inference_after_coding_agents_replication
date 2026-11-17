#!/usr/bin/env python3
"""Score the randomized evaluation's year-two management-safety result."""

import argparse
import json

import pandas as pd
import statsmodels.api as sm


OUTCOME = "mgmt_values_safety_y2"
BASELINE = "mgmt_values_safety_base"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="CSV file with evaluation data")
    parser.add_argument("--output", required=True, help="Path for JSON result")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = [OUTCOME, "treatment", "block"]
    if BASELINE in data.columns:
        required.append(BASELINE)
    sample = data.dropna(subset=required).copy()

    # Block fixed effects implement the within-block randomization adjustment.
    block_indicators = pd.get_dummies(
        sample["block"].astype("category"), drop_first=True, dtype=float
    ).reset_index(drop=True)
    design = pd.concat(
        [sample[["treatment"]].reset_index(drop=True), block_indicators], axis=1
    )
    if BASELINE in required:
        design["baseline_outcome"] = sample[BASELINE].to_numpy()
    design = sm.add_constant(design, has_constant="add")

    result = sm.OLS(sample[OUTCOME].to_numpy(), design).fit(
        cov_type="HC2", use_t=True
    )
    payload = {
        "estimate": float(result.params["treatment"]),
        "std_error": float(result.bse["treatment"]),
        "p_value": float(result.pvalues["treatment"]),
    }
    with open(args.output, "w", encoding="utf-8") as output:
        json.dump(payload, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
