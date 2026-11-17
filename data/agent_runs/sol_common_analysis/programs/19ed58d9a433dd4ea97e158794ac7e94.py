#!/usr/bin/env python3
"""Estimate the intention-to-treat effect on employment at 12 months.

Usage: python score.py --data data.csv --output result.json
Requires numpy, pandas, and statsmodels (with its scipy dependency).
Fits only employed_m12 ~ treatment + site fixed effects, with an intercept,
HC2 standard errors, and a two-sided Student t test with residual degrees
of freedom. Estimates and standard errors use probability units.
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
        parser.error("The output path must differ from the input data path.")

    data = pd.read_csv(args.data, usecols=["site", "treatment", "employed_m12"])
    if data.empty or data.isna().any().any():
        raise ValueError("The analysis requires complete observations.")
    for column in ("treatment", "employed_m12"):
        if not data[column].isin([0, 1]).all():
            raise ValueError(f"{column} must contain only 0 and 1.")

    sites = pd.get_dummies(data["site"], prefix="site", drop_first=True, dtype=float)
    design = sm.add_constant(
        pd.concat([data[["treatment"]], sites], axis=1), has_constant="add"
    ).astype(float)
    if len(data) <= design.shape[1] or np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("The design must have full rank and positive residual degrees of freedom.")

    fit = sm.OLS(data["employed_m12"].astype(float), design).fit(
        cov_type="HC2", use_t=True
    )
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    args.output.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
