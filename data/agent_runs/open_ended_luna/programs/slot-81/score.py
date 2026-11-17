#!/usr/bin/env python3
"""Score the 6-month gender difference in the food-insecurity treatment effect."""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="CSV dataset")
    parser.add_argument("--output", required=True, help="Path for JSON result")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = {"food_insecure_m6", "treatment", "female", "site"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")

    # Separate site intercepts by gender make this equivalent to estimating the
    # treatment effect within each gender (adjusted for site) and contrasting
    # women minus men. HC2 gives the heteroskedasticity-robust covariance.
    fit = smf.ols(
        "food_insecure_m6 ~ treatment * female + C(site) * female", data=data
    ).fit(cov_type="HC2")
    term = "treatment:female"
    result = {
        "estimate": float(fit.params[term]),
        "std_error": float(fit.bse[term]),
        "p_value": float(fit.pvalues[term]),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2, allow_nan=False)
        output_file.write("\n")


if __name__ == "__main__":
    main()
