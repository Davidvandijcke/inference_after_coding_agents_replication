#!/usr/bin/env python3
"""Estimate the intention-to-treat effect on monthly earnings at 12 months.

Usage: python score.py --data data.csv --output result.json
Dependencies: pandas, statsmodels (and their numpy/scipy dependencies).

The only fitted model is earnings_m12 ~ treatment + site indicators.
Inference uses HC2 standard errors and a two-sided Student t test with
n - rank(X) degrees of freedom. The reported p-value is unadjusted; this
program performs no outcome selection or multiple-testing calculations.
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
        parser.error("Data and output paths must differ.")

    data = pd.read_csv(args.data, usecols=["earnings_m12", "treatment", "site"])
    if data.isna().any().any():
        parser.error("The analysis columns must have no missing values.")
    if not np.isfinite(data[["earnings_m12", "treatment"]].to_numpy(dtype=float)).all():
        parser.error("Earnings and treatment must be finite numeric values.")
    if set(data["treatment"].unique()) != {0, 1}:
        parser.error("Treatment must contain both 0 and 1, and no other values.")

    sites = pd.get_dummies(data["site"], prefix="site", drop_first=True, dtype=float)
    design = sm.add_constant(
        pd.concat([data[["treatment"]], sites], axis=1), has_constant="add"
    ).astype(float)
    if len(data) <= design.shape[1] or np.linalg.matrix_rank(design) != design.shape[1]:
        parser.error("The model must have full rank and positive residual degrees of freedom.")

    model = sm.OLS(data["earnings_m12"].astype(float), design).fit(
        cov_type="HC2", use_t=True
    )
    result = {
        "estimate": float(model.params["treatment"]),
        "std_error": float(model.bse["treatment"]),
        "p_value": float(model.pvalues["treatment"]),
    }
    args.output.write_text(json.dumps(result, allow_nan=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
