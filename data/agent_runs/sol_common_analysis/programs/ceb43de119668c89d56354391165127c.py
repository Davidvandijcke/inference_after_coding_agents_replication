#!/usr/bin/env python3
"""Estimate the site-adjusted ITT effect on monthly earnings at 12 months.

Usage: python score.py --data data.csv --output result.json
Requires numpy, pandas, and statsmodels (which depends on scipy).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def score(data_path):
    """Fit only earnings_m12 ~ treatment + site fixed effects, with HC2."""
    data = pd.read_csv(
        data_path, usecols=["site", "treatment", "earnings_m12"]
    )
    if data.empty or data.isna().any().any():
        raise ValueError("The analysis requires nonempty, complete observations.")
    if set(data["treatment"].unique()) != {0, 1}:
        raise ValueError("Treatment must contain both 0 and 1, and no other values.")

    outcome = data["earnings_m12"].astype(float)
    sites = pd.get_dummies(data["site"], prefix="site", drop_first=True, dtype=float)
    design = pd.concat(
        [
            pd.Series(1.0, index=data.index, name="intercept"),
            data[["treatment"]].astype(float),
            sites,
        ],
        axis=1,
    )
    if not np.isfinite(outcome.to_numpy()).all():
        raise ValueError("Earnings must be finite numeric values.")
    if len(data) <= design.shape[1]:
        raise ValueError("Positive residual degrees of freedom are required.")
    if np.linalg.matrix_rank(design.to_numpy()) != design.shape[1]:
        raise ValueError("Treatment and site effects must be identifiable.")

    result = sm.OLS(outcome, design).fit(cov_type="HC2", use_t=True)
    return {
        "estimate": float(result.params["treatment"]),
        "std_error": float(result.bse["treatment"]),
        "p_value": float(result.pvalues["treatment"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error("The output path must differ from the input dataset.")
    values = score(args.data)
    args.output.write_text(json.dumps(values, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
