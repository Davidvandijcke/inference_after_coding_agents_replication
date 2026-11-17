#!/usr/bin/env python3
"""Estimate the 12-month food-insecurity effect among baseline-unemployed participants."""

import argparse
import json
from pathlib import Path

import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV with the evaluation columns")
    parser.add_argument("--output", required=True, help="Path for the JSON result")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = {"employed_baseline", "food_insecure_m12", "treatment", "site"}
    missing = required.difference(data.columns)
    if missing:
        parser.error("Missing required columns: " + ", ".join(sorted(missing)))

    subgroup = data.loc[data["employed_baseline"] == 0]
    if subgroup.empty:
        parser.error("No participants with employed_baseline equal to 0")

    # Site fixed effects; HC2 robust standard error; two-sided t-test.
    fitted = smf.ols(
        "food_insecure_m12 ~ treatment + C(site)", data=subgroup
    ).fit(cov_type="HC2")
    estimate = float(fitted.params["treatment"])
    std_error = float(fitted.bse["treatment"])
    t_stat = estimate / std_error
    p_value = float(2 * stats.t.sf(abs(t_stat), df=fitted.df_resid))

    result = {
        "estimate": estimate,
        "std_error": std_error,
        "p_value": p_value,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2)
        output_file.write("\n")


if __name__ == "__main__":
    main()
