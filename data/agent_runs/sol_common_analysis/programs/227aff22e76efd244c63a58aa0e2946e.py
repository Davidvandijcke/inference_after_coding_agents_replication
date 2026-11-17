#!/usr/bin/env python3
"""Estimate the 12-month employment ITT using site-adjusted OLS and HC2.

Dependencies: pandas, statsmodels.
Usage: python score.py --data data.csv --output result.json
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm


def analyze(data_path):
    data = pd.read_csv(
        data_path, usecols=["site", "treatment", "employed_m12"]
    )
    if data.isna().any().any():
        raise ValueError("The analysis columns must have no missing values.")
    if set(data["treatment"].unique()) != {0, 1}:
        raise ValueError("Treatment must contain both 0 and 1, and no other values.")
    if not data["employed_m12"].isin([0, 1]).all():
        raise ValueError("employed_m12 must be binary (0 or 1).")

    site_indicators = pd.get_dummies(
        data["site"], prefix="site", drop_first=True, dtype=float
    )
    design = sm.add_constant(
        pd.concat([data[["treatment"]], site_indicators], axis=1),
        has_constant="add",
    ).astype(float)
    if len(data) <= design.shape[1]:
        raise ValueError("Insufficient observations for the specified regression.")
    if not data.groupby("site")["treatment"].nunique().gt(1).any():
        raise ValueError("Treatment must vary within at least one site.")

    result = sm.OLS(data["employed_m12"], design).fit(
        cov_type="HC2", use_t=True
    )
    return {
        "estimate": float(result.params["treatment"]),
        "std_error": float(result.bse["treatment"]),
        "p_value": float(result.pvalues["treatment"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = analyze(args.data)
    with args.output.open("w", encoding="utf-8") as output:
        json.dump(result, output, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
