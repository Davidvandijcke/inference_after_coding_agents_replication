#!/usr/bin/env python3
"""Score the selected age 37–49, year two job-search analysis.

Fits OLS among participants aged 37–49 with job_search_likely_y2 observed,
regressing that outcome on treatment and randomization-block indicators.
Reports the treatment coefficient, HC2 standard error, and two-sided t-test
p-value. No search across outcomes or subgroups is performed.
"""

import argparse
import json

import pandas as pd
import statsmodels.api as sm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = {"age_group", "treatment", "block", "job_search_likely_y2"}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    analysis = data.loc[
        data["age_group"].eq(2),
        ["job_search_likely_y2", "treatment", "block"],
    ].dropna()
    if analysis.empty:
        raise ValueError("No observed year two outcomes in age_group 2")

    outcome = analysis["job_search_likely_y2"]
    predictors = pd.get_dummies(
        analysis[["treatment", "block"]],
        columns=["block"],
        drop_first=True,
        dtype=float,
    )
    predictors = sm.add_constant(predictors, has_constant="add")
    fit = sm.OLS(outcome, predictors).fit(cov_type="HC2", use_t=True)

    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2, allow_nan=False)
        output_file.write("\n")


if __name__ == "__main__":
    main()
