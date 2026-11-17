#!/usr/bin/env python3
"""Score the treatment effect on predicted five-year employment at follow-up."""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import t


OUTCOME = "predicted_employment_5y_fu"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="CSV dataset path")
    parser.add_argument("--output", required=True, help="Path for the JSON result")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = {OUTCOME, "treatment", "block"}
    missing = required.difference(data.columns)
    if missing:
        parser.error("data is missing required columns: " + ", ".join(sorted(missing)))

    analysis = data[[OUTCOME, "treatment", "block"]].dropna()
    if analysis.empty or analysis["treatment"].nunique() != 2:
        parser.error("complete cases must include observations in both treatment arms")

    fit = smf.ols(f'{OUTCOME} ~ treatment + C(block)', data=analysis).fit(cov_type="HC2")
    estimate = float(fit.params["treatment"])
    std_error = float(fit.bse["treatment"])
    statistic = estimate / std_error
    p_value = float(2 * t.sf(abs(statistic), fit.df_resid))

    result = {
        "estimate": estimate,
        "std_error": std_error,
        "p_value": p_value,
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
