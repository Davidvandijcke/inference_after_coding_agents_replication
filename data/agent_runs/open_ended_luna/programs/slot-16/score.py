#!/usr/bin/env python3
"""Run the selected Pathways to Work analysis on a compatible CSV dataset."""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


FORMULA = (
    "earnings_m12 ~ treatment + C(site) + age + parent + C(education) + "
    "household_size + employed_baseline + earnings_baseline + "
    "months_unemployed_baseline"
)
REQUIRED_COLUMNS = {
    "earnings_m12",
    "treatment",
    "site",
    "female",
    "age",
    "parent",
    "education",
    "household_size",
    "employed_baseline",
    "earnings_baseline",
    "months_unemployed_baseline",
}


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Estimate the site-adjusted, baseline-adjusted treatment effect "
            "on 12-month earnings among women."
        )
    )
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    missing = sorted(REQUIRED_COLUMNS.difference(data.columns))
    if missing:
        parser.error("input CSV is missing required columns: " + ", ".join(missing))

    women = data.loc[data["female"] == 1]
    if women.empty:
        parser.error("input CSV has no participants with female == 1")

    result = smf.ols(FORMULA, data=women).fit(cov_type="HC2", use_t=True)
    output = {
        "estimate": float(result.params["treatment"]),
        "std_error": float(result.bse["treatment"]),
        "p_value": float(result.pvalues["treatment"]),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
