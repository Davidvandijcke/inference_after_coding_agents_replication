#!/usr/bin/env python3
"""Estimate the assignment effect on 12-month monthly earnings.

Usage: python score.py --data data.csv --output result.json
Requires pandas and statsmodels. Fits OLS with site fixed effects, HC2 standard
errors, and a two-sided Student t test using the residual degrees of freedom.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error("The output path must differ from the input data path.")

    data = pd.read_csv(args.data, usecols=["treatment", "site", "earnings_m12"])
    if data.isna().any().any():
        parser.error("The analysis columns must have no missing values.")
    if set(data["treatment"].unique()) != {0, 1}:
        parser.error("Treatment must contain both 0 and 1, and no other values.")
    if not np.isfinite(data[["treatment", "earnings_m12"]].to_numpy(dtype=float)).all():
        parser.error("Treatment and earnings must be finite numeric values.")

    sites = pd.get_dummies(data["site"], prefix="site", drop_first=True, dtype=float)
    design = sm.add_constant(pd.concat([data[["treatment"]], sites], axis=1))
    matrix = design.to_numpy(dtype=float)
    if len(data) <= matrix.shape[1] or np.linalg.matrix_rank(matrix) < matrix.shape[1]:
        parser.error("The model needs full column rank and positive residual degrees of freedom.")
    fit = sm.OLS(data["earnings_m12"], design).fit(cov_type="HC2", use_t=True)
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
