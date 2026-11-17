#!/usr/bin/env python3
"""Score the block-adjusted year-2 medication-use contrast."""

import argparse
import json

import pandas as pd
import statsmodels.api as sm
from scipy.stats import t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Input CSV dataset")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    needed = ["takes_medication_y2", "takes_medication_base", "treatment", "block"]
    missing = [column for column in needed if column not in data.columns]
    if missing:
        parser.error("missing required columns: " + ", ".join(missing))

    sample = data[needed].dropna()
    x = pd.concat(
        [
            sample[["treatment", "takes_medication_base"]],
            pd.get_dummies(sample["block"].astype("category"), drop_first=True, dtype=float),
        ],
        axis=1,
    )
    x = sm.add_constant(x)
    model = sm.OLS(sample["takes_medication_y2"], x).fit(cov_type="HC2")
    estimate = float(model.params["treatment"])
    std_error = float(model.bse["treatment"])
    p_value = float(2 * t.sf(abs(estimate / std_error), model.df_resid))

    with open(args.output, "w", encoding="utf-8") as output:
        json.dump(
            {"estimate": estimate, "std_error": std_error, "p_value": p_value},
            output,
            indent=2,
        )
        output.write("\n")


if __name__ == "__main__":
    main()
