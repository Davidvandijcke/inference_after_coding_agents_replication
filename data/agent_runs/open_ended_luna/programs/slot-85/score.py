#!/usr/bin/env python3
"""Score the prespecified exploratory gender difference in 12-month earnings."""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


FORMULA = (
    "earnings_m12 ~ treatment * female + C(site) + age + parent + "
    "C(education) + household_size + employed_baseline + earnings_baseline + "
    "months_unemployed_baseline"
)
TEST_TERM = "treatment:female"


def score(data_path: Path) -> dict[str, float]:
    data = pd.read_csv(data_path)
    required = {
        "earnings_m12",
        "treatment",
        "female",
        "site",
        "age",
        "parent",
        "education",
        "household_size",
        "employed_baseline",
        "earnings_baseline",
        "months_unemployed_baseline",
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")

    model = smf.ols(FORMULA, data=data).fit(cov_type="HC2", use_t=True)
    return {
        "estimate": float(model.params[TEST_TERM]),
        "std_error": float(model.bse[TEST_TERM]),
        "p_value": float(model.pvalues[TEST_TERM]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate the site- and baseline-adjusted gender difference in the "
            "program effect on 12-month earnings."
        )
    )
    parser.add_argument("--data", required=True, type=Path, help="Input CSV path")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON path")
    args = parser.parse_args()

    result = score(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
