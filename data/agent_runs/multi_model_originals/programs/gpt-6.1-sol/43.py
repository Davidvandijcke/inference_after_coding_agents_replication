#!/usr/bin/env python3
"""Frozen ITT analysis: six-month weekly hours, adjusted for site only.

Usage: python score.py --data data.csv --output result.json
Dependencies: numpy, pandas, statsmodels (and their dependencies).
Inference: OLS, HC2 standard error, two-sided Student t test with n-rank(X)
degrees of freedom. The p-value is nominal, without search adjustment.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def score(data_path):
    data = pd.read_csv(data_path, usecols=["site", "treatment", "hours_m6"])
    if data.empty or data.isna().any().any():
        raise ValueError("The analysis requires nonempty, complete observations.")
    if set(data["treatment"].unique()) != {0, 1}:
        raise ValueError("Treatment must contain both 0 and 1, and no other values.")
    outcome = pd.to_numeric(data["hours_m6"], errors="raise").astype(float)
    if not np.isfinite(outcome).all():
        raise ValueError("hours_m6 must contain finite numeric values.")
    design = pd.concat(
        [
            data[["treatment"]].astype(float),
            pd.get_dummies(data["site"], prefix="site", drop_first=True, dtype=float),
        ],
        axis=1,
    )
    design = sm.add_constant(design, has_constant="add")
    if len(data) <= design.shape[1] or np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError("The treatment/site model must have full rank and residual degrees of freedom.")
    fitted = sm.OLS(outcome, design).fit(cov_type="HC2", use_t=True)
    result = {
        "estimate": float(fitted.params["treatment"]),
        "std_error": float(fitted.bse["treatment"]),
        "p_value": float(fitted.pvalues["treatment"]),
    }
    if not all(np.isfinite(value) for value in result.values()) or result["std_error"] <= 0:
        raise ValueError("The data do not support finite, nondegenerate HC2 inference.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = score(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
