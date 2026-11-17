#!/usr/bin/env python3
"""Score the treatment effect on WTP sensitivity in the flight-topic block.

Runs OLS of wtp_less_if_ineffective on treatment among block 3 respondents,
using an intercept and HC2 standard errors, then reports the two-sided t-test
for the treatment coefficient.
"""
import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    sample = data.loc[data["block"] == 3, ["treatment", "wtp_less_if_ineffective"]].dropna()
    if sample.empty or sample["treatment"].nunique() != 2:
        raise ValueError("Block 3 must contain nonmissing outcomes in both treatment arms")

    model = smf.ols("wtp_less_if_ineffective ~ treatment", data=sample).fit(cov_type="HC2")
    estimate = float(model.params["treatment"])
    std_error = float(model.bse["treatment"])
    degrees_freedom = float(model.df_resid)
    statistic = estimate / std_error
    p_value = float(2 * t.sf(abs(statistic), degrees_freedom))

    with open(args.output, "w", encoding="utf-8") as output:
        json.dump(
            {"estimate": estimate, "std_error": std_error, "p_value": p_value},
            output,
            indent=2,
        )
        output.write("\n")


if __name__ == "__main__":
    main()
