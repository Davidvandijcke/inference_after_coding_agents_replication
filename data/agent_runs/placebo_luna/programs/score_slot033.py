#!/usr/bin/env python3
"""Score the prespecified exploratory finding in female participants.

Fits job_search_likely_y1 on treatment and randomization-block indicators among
participants with male == 0, using an HC2 robust standard error.
"""

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
    subset = data.loc[data["male"] == 0, ["treatment", "block", "job_search_likely_y1"]].dropna()

    predictors = subset[["treatment"]].astype(float).reset_index(drop=True)
    block_indicators = pd.get_dummies(
        subset["block"].astype("category").reset_index(drop=True),
        drop_first=True,
        dtype=float,
    )
    predictors = pd.concat([predictors, block_indicators], axis=1)
    predictors = sm.add_constant(predictors, has_constant="add")
    outcome = subset["job_search_likely_y1"].astype(float).reset_index(drop=True)

    result = sm.OLS(outcome, predictors).fit(cov_type="HC2", use_t=True)
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
