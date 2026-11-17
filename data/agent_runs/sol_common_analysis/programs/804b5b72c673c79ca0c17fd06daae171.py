#!/usr/bin/env python3
"""Estimate the full-sample effect on employment at 12 months.

Usage: python score.py --data data.csv --output result.json
Requires pandas and statsmodels. Fits only employed_m12 ~ treatment + C(site),
using HC2 standard errors and a two-sided t-test with OLS residual degrees
of freedom. The estimate and standard error are in probability units.
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

    data = pd.read_csv(args.data)
    required = ["employed_m12", "treatment", "site"]
    if data[required].isna().any().any():
        raise ValueError("The analysis columns must have no missing values.")
    for column in ("employed_m12", "treatment"):
        if not data[column].isin([0, 1]).all():
            raise ValueError(f"{column} must be coded 0 or 1.")
    if data["treatment"].nunique() != 2:
        raise ValueError("Both treatment arms are required.")

    fit = smf.ols(
        "employed_m12 ~ treatment + C(site)", data=data, missing="raise"
    ).fit(cov_type="HC2", use_t=True)
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
