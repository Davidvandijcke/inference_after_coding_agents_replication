#!/usr/bin/env python3
"""Estimate the site-adjusted ITT effect on employment at 12 months.

Requires pandas and statsmodels. Outputs the treatment coefficient, HC2
standard error, and unadjusted two-sided t-test p-value. No outcome selection,
subgroup searches, or additional outcome regressions are performed.
"""

import argparse
import json
import math
from pathlib import Path

import pandas as pd
import statsmodels.api as sm


def score(data_path):
    data = pd.read_csv(
        data_path, usecols=["site", "treatment", "employed_m12"]
    )
    if data.isna().any().any():
        raise ValueError("The analysis columns must have no missing values.")
    if set(data["treatment"].unique()) != {0, 1}:
        raise ValueError("Treatment must contain both 0 and 1 only.")
    if not data["employed_m12"].isin([0, 1]).all():
        raise ValueError("Employment at 12 months must be binary (0 or 1).")

    sites = pd.get_dummies(
        data["site"], prefix="site", drop_first=True, dtype=float
    )
    design = sm.add_constant(
        pd.concat([data[["treatment"]], sites], axis=1),
        has_constant="add",
    )
    model = sm.OLS(data["employed_m12"], design, missing="raise").fit(
        cov_type="HC2", use_t=True
    )
    if model.model.rank != design.shape[1] or model.df_resid <= 0:
        raise ValueError("The dataset cannot identify the specified regression.")

    result = {
        "estimate": float(model.params["treatment"]),
        "std_error": float(model.bse["treatment"]),
        "p_value": float(model.pvalues["treatment"]),
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("The dataset does not yield finite inference.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = score(args.data)
    args.output.write_text(json.dumps(result, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
