#!/usr/bin/env python3
"""Score the treatment effect on market-offset reasoning among non-users."""

import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import t as student_t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Input CSV with the evaluation columns")
    parser.add_argument("--output", required=True, help="Path for the JSON result")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = {"uses_wtp_good", "reason_market_offsets", "treatment", "block"}
    missing = required.difference(data.columns)
    if missing:
        parser.error("input is missing required columns: " + ", ".join(sorted(missing)))

    # Restrict to participants who do not consume the good tied to their
    # assigned valuation topic; estimate the treatment effect with block FE.
    sample = data.loc[data["uses_wtp_good"].eq(0)].dropna(
        subset=["reason_market_offsets", "treatment", "block"]
    )
    if sample.empty:
        parser.error("no complete observations in the uses_wtp_good == 0 subgroup")

    fit = smf.ols("reason_market_offsets ~ treatment + C(block)", data=sample).fit(cov_type="HC2")
    estimate = float(fit.params["treatment"])
    std_error = float(fit.bse["treatment"])
    statistic = estimate / std_error
    p_value = float(2 * student_t.sf(abs(statistic), df=fit.df_resid))

    with open(args.output, "w", encoding="utf-8") as output:
        json.dump(
            {"estimate": estimate, "std_error": std_error, "p_value": p_value},
            output,
            indent=2,
        )
        output.write("\n")


if __name__ == "__main__":
    main()
