#!/usr/bin/env python3
"""Score the chosen exploratory analysis: parenthood moderation of
the program effect on debt collections at 6 months.

Fits debt_collections_m6 ~ treatment * parent + site indicators and reports
the HC2-robust, two-sided t-test of the treatment-by-parent interaction.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    data_path = Path(args.data)
    output_path = Path(args.output)
    if data_path.resolve() == output_path.resolve():
        parser.error("--output must not overwrite --data")

    data = pd.read_csv(data_path)
    required = ["site", "treatment", "parent", "debt_collections_m6"]
    missing_columns = [column for column in required if column not in data.columns]
    if missing_columns:
        parser.error("Missing required columns: " + ", ".join(missing_columns))
    if data[required].isna().any().any():
        parser.error("Required analysis columns contain missing values")

    result = smf.ols(
        "debt_collections_m6 ~ treatment * parent + C(site)", data=data
    ).fit(cov_type="HC2", use_t=True)

    interaction = "treatment:parent"
    output = {
        "estimate": float(result.params[interaction]),
        "std_error": float(result.bse[interaction]),
        "p_value": float(result.pvalues[interaction]),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
