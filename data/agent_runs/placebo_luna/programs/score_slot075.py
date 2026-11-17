#!/usr/bin/env python3
"""Run the selected block-adjusted year-1 screening analysis."""

import argparse
import json

import pandas as pd
import statsmodels.api as sm
from scipy.stats import t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = {"block", "treatment", "ever_screened_base", "ever_screened_y1"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    # Outcome ~ treatment + randomization-block indicators + baseline outcome.
    x = pd.concat(
        [
            data[["treatment"]].astype(float),
            pd.get_dummies(
                data["block"].astype("category"),
                prefix="block",
                drop_first=True,
                dtype=float,
            ),
            data[["ever_screened_base"]],
        ],
        axis=1,
    )
    analysis = pd.concat([data[["ever_screened_y1"]], x], axis=1).dropna()
    model = sm.OLS(analysis["ever_screened_y1"], analysis[x.columns]).fit(
        cov_type="HC2", use_t=True
    )

    estimate = float(model.params["treatment"])
    std_error = float(model.bse["treatment"])
    statistic = estimate / std_error
    p_value = float(2 * t.sf(abs(statistic), df=model.df_resid))

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
