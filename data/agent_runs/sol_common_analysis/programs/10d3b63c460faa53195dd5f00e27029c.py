#!/usr/bin/env python3
"""Estimate the intent-to-treat effect on 12-month employment.

Usage: python score.py --data data.csv --output result.json
Requires pandas and statsmodels. Fits only employed_m12 ~ treatment + C(site),
with HC2 standard errors and a two-sided Student t-test using residual df.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm


def score(data_path):
    data = pd.read_csv(data_path, usecols=["site", "treatment", "employed_m12"])
    if data.empty or data.isna().any().any():
        raise ValueError("The analysis requires nonempty, complete observations.")
    for column in ("treatment", "employed_m12"):
        if not data[column].isin([0, 1]).all():
            raise ValueError(f"{column} must be coded 0 or 1.")
    if data["treatment"].nunique() != 2:
        raise ValueError("Both treatment and control observations are required.")

    sites = pd.get_dummies(data["site"], prefix="site", drop_first=True,
                           dtype=float)
    design = sm.add_constant(pd.concat([data[["treatment"]], sites], axis=1),
                             has_constant="add").astype(float)
    model = sm.OLS(data["employed_m12"].astype(float), design).fit(
        cov_type="HC2", use_t=True
    )
    return {
        "estimate": float(model.params["treatment"]),
        "std_error": float(model.bse["treatment"]),
        "p_value": float(model.pvalues["treatment"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = score(args.data)
    if args.data.resolve() == args.output.resolve():
        parser.error("--output must differ from --data")
    args.output.write_text(json.dumps(result, allow_nan=False) + "\n",
                           encoding="utf-8")


if __name__ == "__main__":
    main()
