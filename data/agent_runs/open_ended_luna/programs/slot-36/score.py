#!/usr/bin/env python3
"""Score one preselected treatment-effect contrast for Pathways to Work.

The estimand is the 12-month employment treatment effect for participants
with education below high school minus the effect for all other participants.
The regression includes site indicators and site-by-education-group terms, with
HC2 standard errors and a two-sided t-test.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


REQUIRED_COLUMNS = {"site", "treatment", "education", "employed_m12"}


def score(data_path: Path) -> dict[str, float]:
    data = pd.read_csv(data_path)
    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    if data.empty:
        raise ValueError("The input dataset has no rows")
    if data["education"].isna().any() or data["site"].isna().any():
        raise ValueError("education and site must be observed for every row")
    if data["treatment"].isna().any() or data["employed_m12"].isna().any():
        raise ValueError("treatment and employed_m12 must be observed for every row")
    if not set(data["treatment"].unique()).issubset({0, 1}):
        raise ValueError("treatment must contain only 0 and 1")

    data["low_education"] = (data["education"] == 1).astype(int)
    model = smf.ols(
        "employed_m12 ~ treatment * low_education + C(site) * low_education",
        data=data,
    ).fit(cov_type="HC2", use_t=True)

    term = "treatment:low_education"
    return {
        "estimate": float(model.params[term]),
        "std_error": float(model.bse[term]),
        "p_value": float(model.pvalues[term]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Input CSV path")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON path")
    args = parser.parse_args()

    if args.data.resolve() == args.output.resolve():
        parser.error("--output must not overwrite the input data")

    result = score(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2, allow_nan=False)
        output_file.write("\n")


if __name__ == "__main__":
    main()
