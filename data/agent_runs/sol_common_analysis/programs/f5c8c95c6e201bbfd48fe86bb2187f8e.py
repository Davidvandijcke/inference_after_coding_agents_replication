#!/usr/bin/env python3
"""Estimate the full-sample intention-to-treat effect on 12-month employment.

Usage: python score.py --data data.csv --output result.json
Dependencies: numpy, pandas, statsmodels.
Model: employed_m12 ~ intercept + treatment + site indicators.
Inference: HC2 standard error and a two-sided Student t test, with n - p df.
No outcome selection, subgroup search, or other analyses are performed.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def analyze(data_path):
    data = pd.read_csv(data_path, usecols=["site", "treatment", "employed_m12"])
    if data.empty or data.isna().any().any():
        raise ValueError("The analysis requires nonempty, complete observations.")
    if not data["treatment"].isin([0, 1]).all():
        raise ValueError("treatment must be coded 0 or 1.")
    if not data["employed_m12"].isin([0, 1]).all():
        raise ValueError("employed_m12 must be coded 0 or 1.")

    sites = pd.get_dummies(data["site"], prefix="site", drop_first=True, dtype=float)
    design = pd.concat([data[["treatment"]].astype(float), sites], axis=1)
    design.insert(0, "intercept", 1.0)
    matrix = design.to_numpy()
    if len(data) <= design.shape[1] or np.linalg.matrix_rank(matrix) < design.shape[1]:
        raise ValueError("The regression requires positive residual df and full rank.")

    fit = sm.OLS(data["employed_m12"].astype(float), design).fit(
        cov_type="HC2", use_t=True
    )
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("The regression did not produce finite estimates and inference.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()
    if Path(args.data).resolve() == Path(args.output).resolve():
        parser.error("Input and output paths must differ.")
    result = analyze(args.data)
    Path(args.output).write_text(json.dumps(result, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
