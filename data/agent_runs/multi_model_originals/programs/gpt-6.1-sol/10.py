#!/usr/bin/env python3
"""Estimate the fixed, full-sample 12-month training intention-to-treat effect.

Usage: python score.py --data data.csv --output result.json
Dependencies: numpy, pandas, statsmodels (including scipy).

OLS: training_m12 ~ intercept + treatment + categorical site indicators.
HC2 standard error; two-sided Student t test with n - rank(X) degrees
of freedom. The estimate and standard error use probability units, not
percentage points. The p-value is nominal, without a search adjustment.
No endpoint, subgroup, or model selection is performed.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def analyze(data_path):
    data = pd.read_csv(data_path)
    required = ["treatment", "site", "training_m12"]
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if data[required].isna().any().any():
        raise ValueError("The analysis columns must contain no missing values.")
    for column in ["treatment", "training_m12"]:
        if not data[column].isin([0, 1]).all():
            raise ValueError(f"{column} must contain only 0 and 1.")

    sites = pd.get_dummies(
        data["site"], prefix="site", drop_first=True, dtype=float
    )
    design = sm.add_constant(
        pd.concat([data[["treatment"]], sites], axis=1), has_constant="add"
    ).astype(float)
    if np.linalg.matrix_rank(design.to_numpy()) != design.shape[1]:
        raise ValueError("Treatment and site design matrix must have full rank.")
    if len(data) <= design.shape[1]:
        raise ValueError("Positive residual degrees of freedom are required.")

    result = sm.OLS(data["training_m12"].astype(float), design).fit(
        cov_type="HC2", use_t=True
    )
    output = {
        "estimate": float(result.params["treatment"]),
        "std_error": float(result.bse["treatment"]),
        "p_value": float(result.pvalues["treatment"]),
    }
    if not all(np.isfinite(value) for value in output.values()):
        raise ValueError("The data do not yield finite HC2 inference.")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = analyze(args.data)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
