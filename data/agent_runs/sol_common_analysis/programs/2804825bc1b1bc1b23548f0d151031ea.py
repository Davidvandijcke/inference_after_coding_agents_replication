#!/usr/bin/env python3
"""Estimate the 12-month employment intention-to-treat effect.

Usage: python score.py --data data.csv --output result.json
Requires numpy, pandas, and statsmodels. The sole analysis is OLS of
employed_m12 on treatment, an intercept, and site indicators, with HC2
standard errors and a two-sided Student t test using residual degrees of
freedom. The reported p_value is the unadjusted p-value for this fixed test.
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
        parser.error("--output must differ from --data")

    data = pd.read_csv(args.data, usecols=["site", "treatment", "employed_m12"])
    if data.empty or data.isna().any().any():
        raise ValueError("The chosen analysis requires complete observations.")
    if set(data["treatment"].unique()) != {0, 1}:
        raise ValueError("treatment must contain both 0 and 1.")
    if not data["employed_m12"].isin([0, 1]).all():
        raise ValueError("employed_m12 must contain only 0 and 1.")

    site_indicators = pd.get_dummies(
        data["site"], prefix="site", drop_first=True, dtype=float
    )
    design = pd.concat([data[["treatment"]], site_indicators], axis=1)
    design.insert(0, "intercept", 1.0)
    matrix = design.to_numpy(dtype=float)
    if len(data) <= matrix.shape[1] or np.linalg.matrix_rank(matrix) < matrix.shape[1]:
        raise ValueError("The site-adjusted treatment effect is not estimable.")
    leverage = np.sum(matrix * (matrix @ np.linalg.inv(matrix.T @ matrix)), axis=1)
    if np.any(leverage >= 1.0 - 1e-12):
        raise ValueError("HC2 requires observation leverage strictly below one.")

    fit = sm.OLS(data["employed_m12"], design).fit(cov_type="HC2", use_t=True)
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    output = json.dumps(result, allow_nan=False, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
