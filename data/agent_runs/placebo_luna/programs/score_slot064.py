#!/usr/bin/env python3
"""Score the selected age 31–35 treatment contrast for if_no_norms_post."""

import argparse
import json

import pandas as pd
import statsmodels.api as sm
from scipy import stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="CSV dataset with the evaluation columns")
    parser.add_argument("--output", required=True, help="Path for the JSON result")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    # Age group 2 is the randomized block for participants aged 31–35.
    sample = data.loc[data["block"] == 2, ["treatment", "block", "if_no_norms_post"]].dropna()
    if sample.empty or sample["treatment"].nunique() != 2:
        raise ValueError("The selected subgroup must contain observed outcomes in both arms")

    design = pd.DataFrame(
        {"const": 1.0, "treatment": sample["treatment"].astype(float)},
        index=sample.index,
    )
    block_indicators = pd.get_dummies(sample["block"].astype("category"), drop_first=True, dtype=float)
    design = pd.concat([design, block_indicators], axis=1)

    result = sm.OLS(sample["if_no_norms_post"].astype(float), design).fit(cov_type="HC2")
    estimate = float(result.params["treatment"])
    std_error = float(result.bse["treatment"])
    p_value = float(2 * stats.t.sf(abs(estimate / std_error), result.df_resid))

    with open(args.output, "w", encoding="utf-8") as output:
        json.dump(
            {"estimate": estimate, "std_error": std_error, "p_value": p_value},
            output,
            indent=2,
        )
        output.write("\n")


if __name__ == "__main__":
    main()
