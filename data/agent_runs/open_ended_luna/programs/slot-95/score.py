#!/usr/bin/env python3
"""Score the prespecified education-by-treatment interaction on 6-month employment."""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate the treatment-by-education interaction for 6-month "
            "employment, adjusting for site, with HC2 standard errors."
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
    required = {"site", "treatment", "education", "employed_m6"}
    missing_columns = sorted(required - set(data.columns))
    if missing_columns:
        parser.error(f"input is missing required columns: {', '.join(missing_columns)}")
    if data[list(required)].isna().any().any():
        parser.error("required analysis columns must not contain missing values")

    # The education contrast follows the codebook categories: high school or
    # less (1-2) versus some college or higher (3-4).
    data["college_plus"] = (data["education"] >= 3).astype(int)
    model = smf.ols(
        "employed_m6 ~ treatment * college_plus + C(site)", data=data
    ).fit(cov_type="HC2", use_t=True)

    interaction = "treatment:college_plus"
    result = {
        "estimate": float(model.params[interaction]),
        "std_error": float(model.bse[interaction]),
        "p_value": float(model.pvalues[interaction]),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
