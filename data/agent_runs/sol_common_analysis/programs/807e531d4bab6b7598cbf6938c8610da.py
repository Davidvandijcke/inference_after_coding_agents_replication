#!/usr/bin/env python3
"""Estimate the fixed 12-month employment intention-to-treat effect.

Usage: python score.py --data data.csv --output result.json
Requires pandas and statsmodels. Estimates and standard errors are in
probability units (multiply by 100 for percentage points).
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if args.data.resolve() == args.output.resolve():
        parser.error("--output must differ from --data")

    columns = [
        "employed_m12", "treatment", "site",
        "employed_baseline", "earnings_baseline",
    ]
    data = pd.read_csv(args.data, usecols=columns)
    if data[columns].isna().any().any():
        raise ValueError("The chosen analysis requires complete observations.")
    if set(data["treatment"].unique()) != {0, 1}:
        raise ValueError("Treatment must contain both randomized arms, coded 0 and 1.")

    model = smf.ols(
        "employed_m12 ~ treatment + C(site) + employed_baseline + earnings_baseline",
        data=data,
        missing="raise",
    ).fit(cov_type="HC2", use_t=True)

    result = {
        "estimate": float(model.params["treatment"]),
        "std_error": float(model.bse["treatment"]),
        "p_value": float(model.pvalues["treatment"]),
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
