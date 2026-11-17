#!/usr/bin/env python3
"""Estimate the intention-to-treat effect on 12-month employment.

Usage: python score.py --data data.csv --output result.json
Dependencies: numpy, pandas, statsmodels (and its scipy dependency).
Model: employed_m12 ~ intercept + treatment + site indicators.
Inference: HC2 standard error; two-sided t test with OLS residual degrees
of freedom. The returned p_value is for this single, fixed test.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def estimate_effect(data_path):
    data = pd.read_csv(data_path, usecols=["site", "treatment", "employed_m12"])
    if data.empty or data.isna().any().any():
        raise ValueError("The analysis requires complete observations.")
    for column in ("treatment", "employed_m12"):
        if not data[column].isin([0, 1]).all():
            raise ValueError(f"{column} must contain only 0 and 1.")

    sites = pd.get_dummies(data["site"], prefix="site", drop_first=True, dtype=float)
    design = pd.concat([data[["treatment"]].astype(float), sites], axis=1)
    design = sm.add_constant(design, has_constant="add")
    if len(data) <= design.shape[1] or np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("The regression requires full rank and positive residual degrees of freedom.")

    fitted = sm.OLS(data["employed_m12"].astype(float), design, missing="raise").fit(
        cov_type="HC2", use_t=True
    )
    result = {
        "estimate": float(fitted.params["treatment"]),
        "std_error": float(fitted.bse["treatment"]),
        "p_value": float(fitted.pvalues["treatment"]),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("The data do not support finite HC2 inference.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = estimate_effect(args.data)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
