#!/usr/bin/env python3
"""Estimate the site-adjusted ITT effect on employment at 12 months.

Requires pandas, numpy, and statsmodels. The reported p-value is the nominal
two-sided t-test using HC2 standard errors and OLS residual degrees of freedom.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def score(data_path):
    data = pd.read_csv(data_path, usecols=["site", "treatment", "employed_m12"])
    if data.empty or data.isna().any().any():
        raise ValueError("The analysis requires complete observations.")
    if set(data["treatment"].unique()) != {0, 1}:
        raise ValueError("Treatment must contain both 0 and 1.")
    if not data["employed_m12"].isin([0, 1]).all():
        raise ValueError("Employment must be coded 0 or 1.")

    sites = pd.get_dummies(data["site"], prefix="site", drop_first=True,
                           dtype=float)
    design = pd.concat([data[["treatment"]].astype(float), sites], axis=1)
    design.insert(0, "intercept", 1.0)
    matrix = design.to_numpy()
    if len(data) <= matrix.shape[1] or np.linalg.matrix_rank(matrix) < matrix.shape[1]:
        raise ValueError("The regression needs positive residual degrees of freedom and full rank.")

    fit = sm.OLS(data["employed_m12"].astype(float), design).fit(
        cov_type="HC2", use_t=True
    )
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("The regression did not produce finite inference.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error("The output must not overwrite the input data.")
    result = score(args.data)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
