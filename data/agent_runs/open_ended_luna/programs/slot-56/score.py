#!/usr/bin/env python3
"""Score the treatment-effect difference by baseline employment for food insecurity."""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate whether the randomized program effect on 12-month food "
            "insecurity differs by baseline employment."
        )
    )
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    data_path = Path(args.data)
    output_path = Path(args.output)
    if data_path.resolve() == output_path.resolve():
        parser.error("--output must not overwrite --data")

    data = pd.read_csv(data_path)
    required = ["food_insecure_m12", "treatment", "employed_baseline", "site"]
    missing_columns = [column for column in required if column not in data.columns]
    if missing_columns:
        parser.error("Missing required columns: " + ", ".join(missing_columns))
    if data[required].isna().any().any():
        parser.error("Required analysis columns must not contain missing values")
    for column in ["treatment", "employed_baseline"]:
        if not set(data[column].unique()).issubset({0, 1}):
            parser.error(f"{column} must be coded 0/1")

    # Site indicators reflect the within-site randomization. The interaction
    # coefficient is the difference in treatment effects by baseline status.
    model = smf.ols(
        "food_insecure_m12 ~ treatment * employed_baseline + C(site)",
        data=data,
    ).fit(cov_type="HC2", use_t=True)
    term = "treatment:employed_baseline"

    result = {
        "estimate": float(model.params[term]),
        "std_error": float(model.bse[term]),
        "p_value": float(model.pvalues[term]),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
