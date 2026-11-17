#!/usr/bin/env python3
"""Score one baseline-adjusted, site-stratified treatment effect on 6-month earnings."""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


FORMULA = (
    "earnings_m6 ~ treatment + C(site) + age + female + parent + C(education) + "
    "household_size + employed_baseline + earnings_baseline + "
    "months_unemployed_baseline"
)
REQUIRED_COLUMNS = [
    "earnings_m6",
    "treatment",
    "site",
    "age",
    "female",
    "parent",
    "education",
    "household_size",
    "employed_baseline",
    "earnings_baseline",
    "months_unemployed_baseline",
]


def main():
    parser = argparse.ArgumentParser(
        description="Estimate the selected Pathways to Work earnings effect."
    )
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {', '.join(missing)}")
    if data.empty:
        raise ValueError("Input CSV has no observations")
    if data[REQUIRED_COLUMNS].isna().any().any():
        raise ValueError("Analysis columns must not contain missing values")

    result = smf.ols(FORMULA, data=data).fit(cov_type="HC2", use_t=True)
    output = {
        "estimate": float(result.params["treatment"]),
        "std_error": float(result.bse["treatment"]),
        "p_value": float(result.pvalues["treatment"]),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, allow_nan=False)
        file.write("\n")


if __name__ == "__main__":
    main()
