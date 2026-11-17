#!/usr/bin/env python3
"""Score the 12-month earnings interaction by baseline employment status."""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate whether the program's 12-month earnings effect differs "
            "by baseline employment status."
        )
    )
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    needed = ["earnings_m12", "treatment", "employed_baseline", "site"]
    missing = [column for column in needed if column not in data.columns]
    if missing:
        parser.error("input is missing required columns: " + ", ".join(missing))
    data = data[needed].dropna()
    if data.empty:
        parser.error("input has no complete rows for the chosen analysis")

    # Site-by-baseline-status intercepts reproduce site-adjusted estimates
    # within each baseline-employment subgroup. The interaction coefficient
    # is the employed subgroup effect minus the not-employed subgroup effect.
    model = smf.ols(
        "earnings_m12 ~ C(site) * C(employed_baseline) "
        "+ treatment * C(employed_baseline)",
        data=data,
    ).fit(cov_type="HC2", use_t=True)
    term = "treatment:C(employed_baseline)[T.1]"
    if term not in model.params:
        parser.error("baseline employment must include both 0 and 1")

    result = {
        "estimate": float(model.params[term]),
        "std_error": float(model.bse[term]),
        "p_value": float(model.pvalues[term]),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
