#!/usr/bin/env python3
"""Estimate the 12-month employment ITT using site OLS and HC2 inference.

Dependencies: numpy, pandas, statsmodels.
Usage: python score.py --data data.csv --output result.json
The reported p_value is the unadjusted, two-sided treatment t-test.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def analyze(data_path):
    data = pd.read_csv(data_path, usecols=["site", "treatment", "employed_m12"])
    if data.isna().any().any():
        raise ValueError("The analysis columns must have no missing values.")
    if not data["treatment"].isin([0, 1]).all():
        raise ValueError("treatment must be coded 0 or 1.")
    if not data["employed_m12"].isin([0, 1]).all():
        raise ValueError("employed_m12 must be coded 0 or 1.")

    sites = pd.get_dummies(data["site"], prefix="site", drop_first=True, dtype=float)
    design = sm.add_constant(
        pd.concat([data[["treatment"]], sites], axis=1).astype(float),
        has_constant="add",
    )
    if len(data) <= design.shape[1] or np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError("The site-adjusted model needs positive residual degrees of freedom and full rank.")

    fit = sm.OLS(data["employed_m12"].astype(float), design).fit(
        cov_type="HC2", use_t=True
    )
    return {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input participant CSV")
    parser.add_argument("--output", required=True, help="Output JSON file")
    args = parser.parse_args()
    if Path(args.data).resolve() == Path(args.output).resolve():
        parser.error("The output path must differ from the input data path.")
    result = analyze(args.data)
    Path(args.output).write_text(json.dumps(result, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
