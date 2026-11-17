#!/usr/bin/env python3
"""Estimate the 12-month employment intention-to-treat effect.

Usage: python score.py --data data.csv --output result.json
Requires pandas, numpy, and statsmodels (including scipy).

Fits OLS: employed_m12 ~ treatment + site indicators, with an intercept.
Returns the treatment coefficient, HC2 standard error, and unadjusted
two-sided Student t-test p-value using the regression residual degrees
of freedom. No outcome screening or other analyses are performed.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def score(data_path):
    data = pd.read_csv(data_path, usecols=["employed_m12", "treatment", "site"])
    if data.empty or data.isna().any().any():
        raise ValueError("The analysis columns must be nonempty and complete.")
    for column in ("employed_m12", "treatment"):
        if not data[column].isin([0, 1]).all():
            raise ValueError(f"{column} must contain only 0 and 1.")

    sites = pd.get_dummies(data["site"], prefix="site", drop_first=True,
                           dtype=float)
    design = sm.add_constant(
        pd.concat([data[["treatment"]].astype(float), sites], axis=1),
        has_constant="add",
    )
    if np.linalg.matrix_rank(design.to_numpy()) != design.shape[1]:
        raise ValueError("The treatment effect is not identified by this design.")
    if len(data) <= design.shape[1]:
        raise ValueError("Positive residual degrees of freedom are required.")

    fit = sm.OLS(data["employed_m12"].astype(float), design).fit(
        cov_type="HC2", use_t=True
    )
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("The data do not support finite HC2 inference.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = score(args.data)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
