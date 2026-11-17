#!/usr/bin/env python3
"""Estimate the site-adjusted ITT effect on 12-month employment.

Usage: python score.py --data data.csv --output result.json
Requires numpy, pandas, scipy, and statsmodels.
The estimate and standard error are in probability units, not percentage points.
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

    data = pd.read_csv(
        args.data, usecols=["site", "treatment", "employed_m12"]
    )
    if data.empty or data.isna().any().any():
        raise ValueError("The analysis requires nonempty, complete data.")
    for column in ("treatment", "employed_m12"):
        if not data[column].isin([0, 1]).all():
            raise ValueError(f"{column} must contain only 0 and 1.")

    sites = pd.get_dummies(data["site"], prefix="site", drop_first=True,
                           dtype=float)
    design = sm.add_constant(
        pd.concat([data[["treatment"]], sites], axis=1),
        has_constant="add",
    ).astype(float)
    if len(data) <= design.shape[1] or np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("The specified regression needs a full-rank design and residual degrees of freedom.")

    fit = sm.OLS(data["employed_m12"].astype(float), design).fit(
        cov_type="HC2", use_t=True
    )
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
