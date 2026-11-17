#!/usr/bin/env python3
"""Estimate the assignment effect on 12-month food insecurity.

Usage: python score.py --data data.csv --output result.json
Requires pandas and statsmodels. Fits only OLS of food_insecure_m12 on
treatment and site fixed effects, with HC2 covariance and a two-sided
Student t test using residual degrees of freedom. The p-value is nominal.
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

    data = pd.read_csv(args.data)
    required = ["food_insecure_m12", "treatment", "site"]
    missing = set(required).difference(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if data[required].isna().any().any():
        raise ValueError("Analysis columns must be fully observed.")
    for column in ["food_insecure_m12", "treatment"]:
        if not data[column].isin([0, 1]).all():
            raise ValueError(f"{column} must contain only 0 and 1.")
    if data["treatment"].nunique() != 2:
        raise ValueError("Both assignment groups must be present.")

    model = smf.ols(
        "food_insecure_m12 ~ treatment + C(site)", data=data, missing="raise"
    ).fit(cov_type="HC2", use_t=True)
    result = {
        "estimate": float(model.params["treatment"]),
        "std_error": float(model.bse["treatment"]),
        "p_value": float(model.pvalues["treatment"]),
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
