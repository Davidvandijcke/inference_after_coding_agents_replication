#!/usr/bin/env python3
"""Score the prespecified lead analysis on a dataset with the packet columns."""

import argparse
import json

import pandas as pd
import statsmodels.api as sm


OUTCOME = "mgmt_values_safety_y2"
BASELINE = "mgmt_values_safety_base"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    required = ["treatment", "block", OUTCOME, BASELINE]
    missing = [column for column in required if column not in df.columns]
    if missing:
        parser.error("data is missing required columns: " + ", ".join(missing))

    analysis = df[required].dropna()
    predictors = pd.concat(
        [
            analysis[["treatment", BASELINE]].astype(float),
            pd.get_dummies(analysis["block"].astype("category"), drop_first=True, dtype=float),
        ],
        axis=1,
    )
    predictors = sm.add_constant(predictors)
    result = sm.OLS(analysis[OUTCOME].astype(float), predictors).fit(
        cov_type="HC2", use_t=True
    )
    output = {
        "estimate": float(result.params["treatment"]),
        "std_error": float(result.bse["treatment"]),
        "p_value": float(result.pvalues["treatment"]),
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
