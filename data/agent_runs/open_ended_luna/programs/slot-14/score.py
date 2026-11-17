#!/usr/bin/env python3
"""Score the selected site 1, 12-month self-rated health contrast."""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Estimate the treatment effect on health_m12 among site 1 "
            "participants using OLS with HC2 standard errors."
        )
    )
    parser.add_argument("--data", required=True, help="Path to input CSV")
    parser.add_argument("--output", required=True, help="Path to output JSON")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = {"site", "treatment", "health_m12"}
    missing = required.difference(data.columns)
    if missing:
        parser.error("input is missing columns: " + ", ".join(sorted(missing)))

    sample = data.loc[data["site"] == 1, ["treatment", "health_m12"]].dropna()
    if sample.empty or sample["treatment"].nunique() != 2:
        parser.error("site 1 must contain observed outcomes in both treatment arms")

    y = sample["health_m12"].to_numpy(dtype=float)
    x = sm.add_constant(sample["treatment"].to_numpy(dtype=float))
    result = sm.OLS(y, x).fit(cov_type="HC2", use_t=True)

    output = {
        "estimate": float(result.params[1]),
        "std_error": float(result.bse[1]),
        "p_value": float(result.pvalues[1]),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
