#!/usr/bin/env python3
"""Score the treatment effect on market-offset explanations among independents."""

import argparse
import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    needed = {"politics", "treatment", "block", "reason_market_offsets"}
    missing = needed.difference(data.columns)
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))

    # This is the one selected analysis: political independents (politics == 3).
    sample = data.loc[data["politics"] == 3].dropna(
        subset=["treatment", "block", "reason_market_offsets"]
    )
    if sample.empty or sample["treatment"].nunique() != 2:
        raise ValueError("Selected subgroup must contain both treatment groups")

    block_indicators = pd.get_dummies(
        sample["block"].astype("category"), drop_first=True, dtype=float
    )
    design = pd.concat(
        [sample[["treatment"]].astype(float), block_indicators], axis=1
    )
    design = sm.add_constant(design, has_constant="add")
    model = sm.OLS(sample["reason_market_offsets"].astype(float), design).fit(
        cov_type="HC2"
    )

    estimate = float(model.params["treatment"])
    std_error = float(model.bse["treatment"])
    statistic = estimate / std_error
    p_value = float(2 * stats.t.sf(abs(statistic), df=model.df_resid))

    with open(args.output, "w", encoding="utf-8") as output:
        json.dump(
            {
                "estimate": estimate,
                "std_error": std_error,
                "p_value": p_value,
            },
            output,
            indent=2,
        )
        output.write("\n")


if __name__ == "__main__":
    main()
