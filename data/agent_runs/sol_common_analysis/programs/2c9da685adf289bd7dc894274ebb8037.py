#!/usr/bin/env python3
"""Estimate the six-month employment ITT effect with site fixed effects.

Usage: python score.py --data data.csv --output result.json
Requires numpy, pandas, and statsmodels.
The p-value is the unadjusted, two-sided HC2 t-test with OLS residual df.
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
        parser.error("The output must be separate from the input data.")

    data = pd.read_csv(args.data, usecols=["site", "treatment", "employed_m6"])
    if data.isna().any().any():
        raise ValueError("The analysis columns must have no missing values.")
    if not data["treatment"].isin([0, 1]).all():
        raise ValueError("treatment must be coded 0 or 1.")
    if not data["employed_m6"].isin([0, 1]).all():
        raise ValueError("employed_m6 must be coded 0 or 1.")

    sites = pd.get_dummies(data["site"], prefix="site", drop_first=True, dtype=float)
    design = pd.concat([data[["treatment"]].astype(float), sites], axis=1)
    design = sm.add_constant(design, has_constant="add")
    if len(data) <= design.shape[1] or np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("The regression requires full rank and positive residual df.")

    fit = sm.OLS(data["employed_m6"].astype(float), design).fit(
        cov_type="HC2", use_t=True
    )
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("The regression did not produce finite inference results.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
