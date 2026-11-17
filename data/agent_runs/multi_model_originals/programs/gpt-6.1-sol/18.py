#!/usr/bin/env python3
"""Estimate the fixed, full-sample effect on 12-month psychological distress.

Model: distress_m12 ~ intercept + treatment + categorical site indicators.
Inference: HC2 standard error and two-sided Student t test, with residual
degrees of freedom n - rank(X). No subgroup, outcome, or model search is run.

Dependencies: numpy, pandas, scipy, statsmodels.
Usage: python score.py --data data.csv --output result.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def analyze(data_path):
    data = pd.read_csv(data_path)
    required = ["site", "treatment", "distress_m12"]
    missing = set(required).difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if data[required].isna().any().any():
        raise ValueError("Required analysis columns must have no missing values.")
    if set(data["treatment"].unique()) != {0, 1}:
        raise ValueError("Treatment must contain both 0 and 1, and no other values.")

    sites = pd.get_dummies(data["site"], prefix="site", drop_first=True,
                           dtype=float)
    design = pd.concat([data[["treatment"]], sites], axis=1)
    design.insert(0, "intercept", 1.0)
    design = design.astype(float)
    outcome = pd.to_numeric(data["distress_m12"], errors="raise").astype(float)
    if not np.isfinite(outcome).all():
        raise ValueError("distress_m12 must contain only finite values.")
    rank = np.linalg.matrix_rank(design.to_numpy())
    if rank != design.shape[1] or len(data) <= rank:
        raise ValueError("The model requires a full-rank design and residual degrees of freedom.")

    fitted = sm.OLS(outcome, design).fit(cov_type="HC2", use_t=True)
    result = {
        "estimate": float(fitted.params["treatment"]),
        "std_error": float(fitted.bse["treatment"]),
        "p_value": float(fitted.pvalues["treatment"]),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("The fitted model produced non-finite inference results.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error("The output path must differ from the input data path.")
    result = analyze(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8")


if __name__ == "__main__":
    main()
