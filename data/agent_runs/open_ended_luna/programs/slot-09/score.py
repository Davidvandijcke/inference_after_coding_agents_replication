#!/usr/bin/env python3
"""Score the selected Pathways to Work subgroup analysis.

This runs only the exploratory analysis selected from the supplied evaluation:
12-month earnings regressed on treatment and site indicators among participants
younger than 35. It does not repeat the outcome or subgroup search.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV file")
    parser.add_argument("--output", required=True, help="Output JSON file")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = {"age", "site", "treatment", "earnings_m12"}
    missing = sorted(required.difference(data.columns))
    if missing:
        parser.error("input CSV is missing required columns: " + ", ".join(missing))

    subgroup = data.loc[data["age"] < 35].copy()
    if subgroup.empty or subgroup["treatment"].nunique() != 2:
        parser.error("the under-35 subgroup must contain both treatment arms")

    result = smf.ols(
        "earnings_m12 ~ treatment + C(site)", data=subgroup
    ).fit(cov_type="HC2", use_t=True)

    payload = {
        "estimate": float(result.params["treatment"]),
        "std_error": float(result.bse["treatment"]),
        "p_value": float(result.pvalues["treatment"]),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2)
        output_file.write("\n")


if __name__ == "__main__":
    main()
