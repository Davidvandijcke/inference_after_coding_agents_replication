#!/usr/bin/env python3
"""Estimate the fixed six-month training-enrollment intention-to-treat effect.

Usage: python score.py --data data.csv --output result.json
Requires: pandas, numpy, scipy, statsmodels.

Model: training_m6 ~ intercept + treatment + categorical site indicators.
Inference: HC2 standard error and two-sided t test with residual degrees of
freedom. The JSON p_value is nominal, without a multiple-testing adjustment.
No outcome selection or additional analyses are performed by this program.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def score(data_path):
    data = pd.read_csv(data_path)
    required = ["training_m6", "treatment", "site"]
    missing = set(required).difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    if data[required].isna().any().any():
        raise ValueError("The analysis columns must be fully observed.")
    if set(data["treatment"].unique()) != {0, 1}:
        raise ValueError("Treatment must contain both 0 and 1, and no other values.")
    if not data["training_m6"].isin([0, 1]).all():
        raise ValueError("training_m6 must be binary (0 or 1).")

    sites = pd.get_dummies(data["site"], prefix="site", drop_first=True, dtype=float)
    design = pd.concat([data[["treatment"]].astype(float), sites], axis=1)
    design = sm.add_constant(design, has_constant="add")
    if len(data) <= design.shape[1] or np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError("The model needs full column rank and positive residual degrees of freedom.")

    fit = sm.OLS(data["training_m6"].astype(float), design).fit(
        cov_type="HC2", use_t=True
    )
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("The requested model did not produce finite inference.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error("Input and output paths must differ.")
    result = score(args.data)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
