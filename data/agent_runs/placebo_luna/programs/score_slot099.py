#!/usr/bin/env python3
"""Score the year-2 any-sick-days treatment effect in a supplied trial dataset."""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
from scipy import stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV with trial data")
    parser.add_argument("--output", required=True, help="Path for JSON result")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = ["block", "treatment", "any_sick_days_base", "any_sick_days_y2"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        parser.error("input is missing required columns: " + ", ".join(missing))

    covariates = pd.concat(
        [
            data["treatment"].rename("treatment"),
            pd.get_dummies(data["block"], prefix="block", drop_first=True, dtype=float),
            data["any_sick_days_base"].rename("baseline"),
        ],
        axis=1,
    ).astype(float)
    analysis = pd.concat([data["any_sick_days_y2"], covariates], axis=1).dropna()
    model = sm.OLS(
        analysis["any_sick_days_y2"], sm.add_constant(analysis[covariates.columns])
    ).fit(cov_type="HC2")

    estimate = float(model.params["treatment"])
    std_error = float(model.bse["treatment"])
    statistic = estimate / std_error
    p_value = float(2 * stats.t.sf(abs(statistic), model.df_resid))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {"estimate": estimate, "std_error": std_error, "p_value": p_value},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
