#!/usr/bin/env python3
"""Estimate the assignment effect on 12-month employment.

Usage: python score.py --data data.csv --output result.json
Dependencies: pandas, statsmodels (and their dependencies).
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    columns = ["site", "treatment", "employed_baseline", "employed_m12"]
    data = pd.read_csv(args.data, usecols=columns)
    if data.empty or data[columns].isna().any().any():
        raise ValueError("The analysis requires nonempty, complete observations.")
    for column in ["treatment", "employed_baseline", "employed_m12"]:
        if not data[column].isin([0, 1]).all():
            raise ValueError(f"{column} must contain only 0 and 1.")
    if data["treatment"].nunique() != 2:
        raise ValueError("Both randomized treatment groups are required.")

    design = pd.concat(
        [
            data[["treatment", "employed_baseline"]].astype(float),
            pd.get_dummies(data["site"], prefix="site", drop_first=True, dtype=float),
        ],
        axis=1,
    )
    design = sm.add_constant(design, has_constant="add")
    model = sm.OLS(data["employed_m12"], design, missing="raise").fit(
        cov_type="HC2", use_t=True
    )
    result = {
        "estimate": float(model.params["treatment"]),
        "std_error": float(model.bse["treatment"]),
        "p_value": float(model.pvalues["treatment"]),
    }
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
