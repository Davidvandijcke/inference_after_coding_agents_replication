#!/usr/bin/env python3
"""Estimate the intention-to-treat effect on monthly earnings at 12 months.

Usage: python score.py --data data.csv --output result.json
Requires: numpy, pandas, scipy, statsmodels.
Model: earnings_m12 ~ intercept + treatment + site indicators.
Inference: HC2 standard error and two-sided Student t test, using n - rank(X)
residual degrees of freedom. No other outcome or specification is analyzed.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def score(data_path):
    data = pd.read_csv(data_path, usecols=["site", "treatment", "earnings_m12"])
    if data.empty or data.isna().any().any():
        raise ValueError("The selected analysis requires complete observations.")
    if set(data["treatment"].unique()) != {0, 1}:
        raise ValueError("Treatment must contain both 0 and 1, and no other values.")
    if not np.isfinite(data[["treatment", "earnings_m12"]].to_numpy(dtype=float)).all():
        raise ValueError("Treatment and earnings must be finite numeric values.")

    sites = pd.get_dummies(data["site"], prefix="site", drop_first=True, dtype=float)
    design = pd.concat([data[["treatment"]].astype(float), sites], axis=1)
    design = sm.add_constant(design, has_constant="add")
    matrix = design.to_numpy()
    if len(data) <= matrix.shape[1] or np.linalg.matrix_rank(matrix) != matrix.shape[1]:
        raise ValueError("The model requires full rank and positive residual degrees of freedom.")
    leverage = np.sum((matrix @ np.linalg.pinv(matrix.T @ matrix)) * matrix, axis=1)
    if np.any(leverage >= 1 - 1e-12):
        raise ValueError("HC2 is undefined for observations with leverage equal to one.")

    fit = sm.OLS(data["earnings_m12"].astype(float), design).fit(cov_type="HC2", use_t=True)
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("The selected analysis did not produce finite statistics.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV with the codebook columns")
    parser.add_argument("--output", required=True, help="Destination JSON file")
    args = parser.parse_args()
    result = score(args.data)
    Path(args.output).write_text(json.dumps(result, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
