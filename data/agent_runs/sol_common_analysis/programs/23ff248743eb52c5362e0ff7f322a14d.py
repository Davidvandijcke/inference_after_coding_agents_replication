#!/usr/bin/env python3
"""Estimate the intention-to-treat effect on employment at 12 months.

Usage: python score.py --data data.csv --output result.json
Requires pandas and statsmodels. The fixed analysis is OLS of employed_m12
on an intercept, treatment, and site indicators, with HC2 standard errors
and a two-sided Student t test using residual degrees of freedom.
The reported p-value is nominal (not adjusted for the exploratory search).
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm


def score(data_path):
    data = pd.read_csv(
        data_path, usecols=["site", "treatment", "employed_m12"]
    )
    if data.isna().any().any():
        raise ValueError("The analysis columns must be fully observed.")
    if set(data["treatment"].unique()) != {0, 1}:
        raise ValueError("Treatment must contain both 0 and 1.")

    site_indicators = pd.get_dummies(
        data["site"], prefix="site", drop_first=True, dtype=float
    )
    design = sm.add_constant(
        pd.concat([data[["treatment"]], site_indicators], axis=1),
        has_constant="add",
    )
    fit = sm.OLS(data["employed_m12"], design, missing="raise").fit(
        cov_type="HC2", use_t=True
    )
    return {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = score(args.data)
    args.output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
