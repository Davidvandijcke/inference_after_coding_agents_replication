#!/usr/bin/env python3
"""Estimate the full-sample effect of assignment on 12-month employment.

Usage: python score.py --data data.csv --output result.json
Requires pandas and statsmodels.

The sole model is OLS: employed_m12 ~ treatment + C(site) +
employed_baseline. Inference uses HC2 standard errors and a two-sided
Student t-test with the OLS residual degrees of freedom.
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

    columns = ["employed_m12", "treatment", "site", "employed_baseline"]
    data = pd.read_csv(args.data, usecols=columns)
    if data.empty or data.isna().any().any():
        raise ValueError("The analysis requires complete, nonempty data.")
    for column in ["employed_m12", "treatment", "employed_baseline"]:
        if not data[column].isin([0, 1]).all():
            raise ValueError(f"{column} must contain only 0 and 1.")
    if data["treatment"].nunique() != 2:
        raise ValueError("Both treatment groups must be present.")

    model = smf.ols(
        "employed_m12 ~ treatment + C(site) + employed_baseline",
        data=data,
        missing="raise",
    ).fit(cov_type="HC2", use_t=True)

    result = {
        "estimate": float(model.params["treatment"]),
        "std_error": float(model.bse["treatment"]),
        "p_value": float(model.pvalues["treatment"]),
    }
    args.output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
