#!/usr/bin/env python3
"""Score the selected exploratory analysis on a matching evaluation dataset."""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate the block-adjusted, baseline-adjusted treatment effect on "
            "year 1 physical problems among male participants."
        )
    )
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    outcome = "physical_problems_y1"
    baseline = "physical_problems_base"
    required = ["male", "treatment", "block", baseline, outcome]
    data = pd.read_csv(args.data)
    missing = [column for column in required if column not in data.columns]
    if missing:
        parser.error(f"input CSV is missing required columns: {', '.join(missing)}")

    sample = data.loc[data["male"] == 1, required].dropna()
    if sample.empty:
        parser.error("no complete male participant observations for the selected analysis")

    fit = smf.ols(
        f"{outcome} ~ treatment + {baseline} + C(block)", data=sample
    ).fit(cov_type="HC2", use_t=True)

    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
