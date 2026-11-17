#!/usr/bin/env python3
"""Score the preselected site 2 six-month employment analysis."""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


FORMULA = (
    "employed_m6 ~ treatment + age + female + parent + C(education) "
    "+ household_size + employed_baseline + earnings_baseline "
    "+ months_unemployed_baseline"
)
REQUIRED_COLUMNS = {
    "site",
    "treatment",
    "employed_m6",
    "age",
    "female",
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
            "Estimate the baseline-adjusted treatment effect on six-month "
            "employment among participants at site 2."
        )
    )
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    missing = sorted(REQUIRED_COLUMNS - set(data.columns))
    if missing:
        parser.error("input CSV is missing required columns: " + ", ".join(missing))

    site_data = data.loc[data["site"] == 2].copy()
    if site_data.empty or site_data["treatment"].nunique() != 2:
        parser.error("site 2 must contain participants in both treatment arms")

    fitted = smf.ols(FORMULA, data=site_data).fit()
    result = fitted.get_robustcov_results(cov_type="HC2", use_t=True)
    treatment_index = fitted.model.exog_names.index("treatment")

    output = {
        "estimate": float(result.params[treatment_index]),
        "std_error": float(result.bse[treatment_index]),
        "p_value": float(result.pvalues[treatment_index]),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
