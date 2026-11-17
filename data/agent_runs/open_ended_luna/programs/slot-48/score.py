#!/usr/bin/env python3
"""Score the 12-month food-insecurity treatment effect."""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV with evaluation data")
    parser.add_argument("--output", required=True, help="Path for the JSON result")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = ["food_insecure_m12", "treatment", "site"]
    missing = [name for name in required if name not in data.columns]
    if missing:
        parser.error("CSV is missing required columns: " + ", ".join(missing))
    if data[required].isna().any().any():
        parser.error("Required analysis columns contain missing values")

    # Site fixed effects; HC2 robust standard error; two-sided t-test.
    result = smf.ols(
        "food_insecure_m12 ~ treatment + C(site)", data=data
    ).fit(cov_type="HC2", use_t=True)

    output = {
        "estimate": float(result.params["treatment"]),
        "std_error": float(result.bse["treatment"]),
        "p_value": float(result.pvalues["treatment"]),
    }
    Path(args.output).write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
