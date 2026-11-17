#!/usr/bin/env python3
"""Estimate the 12-month employment intention-to-treat effect.

Usage: python score.py --data data.csv --output result.json
Requires: numpy, pandas, statsmodels.
The p_value is the unadjusted, two-sided HC2 t-test p-value.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def score(data_path):
    """Run only employed_m12 ~ treatment + categorical site indicators."""
    data = pd.read_csv(data_path, usecols=["treatment", "site", "employed_m12"])
    if data.empty or data.isna().any().any():
        raise ValueError("The analysis requires nonempty, complete observations.")
    for column in ("treatment", "employed_m12"):
        if not data[column].isin([0, 1]).all():
            raise ValueError(f"{column} must contain only 0 and 1.")

    site_indicators = pd.get_dummies(
        data["site"], prefix="site", drop_first=True, dtype=float
    )
    design = sm.add_constant(
        pd.concat([data[["treatment"]].astype(float), site_indicators], axis=1),
        has_constant="add",
    )
    matrix = design.to_numpy()
    if len(data) <= design.shape[1] or np.linalg.matrix_rank(matrix) < design.shape[1]:
        raise ValueError("The site-adjusted model must have full rank and residual degrees of freedom.")

    fit = sm.OLS(data["employed_m12"].astype(float), design).fit(
        cov_type="HC2", use_t=True
    )
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("The model did not produce finite inference results.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error("--output must differ from --data")
    result = score(args.data)
    with args.output.open("w", encoding="utf-8") as output:
        json.dump(result, output, allow_nan=False, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
