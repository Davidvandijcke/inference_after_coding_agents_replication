#!/usr/bin/env python3
"""Estimate the intent-to-treat effect on employment at 12 months.

Usage: python score.py --data data.csv --output result.json
Dependencies: numpy, pandas, statsmodels.
The estimate and standard error use probability units (1 = 100 percentage
points). The p-value is the unadjusted, two-sided HC2 t-test p-value.
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
        parser.error("Input must contain complete observations")
    if set(data["treatment"].unique()) != {0, 1}:
        parser.error("treatment must contain both 0 and 1")
    if not data["employed_m12"].isin([0, 1]).all():
        parser.error("employed_m12 must contain only 0 and 1")

    # An intercept and one omitted site give the site fixed-effects model.
    sites = pd.get_dummies(data["site"], prefix="site", drop_first=True,
                           dtype=float)
    design = sm.add_constant(pd.concat([data[["treatment"]], sites], axis=1),
                             has_constant="add").astype(float)
    if len(data) <= design.shape[1]:
        parser.error("Insufficient residual degrees of freedom")
    if np.linalg.matrix_rank(design.to_numpy()) != design.shape[1]:
        parser.error("Treatment and site coefficients are not identified")

    model = sm.OLS(data["employed_m12"].astype(float), design).fit(
        cov_type="HC2", use_t=True
    )
    result = {
        "estimate": float(model.params["treatment"]),
        "std_error": float(model.bse["treatment"]),
        "p_value": float(model.pvalues["treatment"]),
    }
    if not all(np.isfinite(value) for value in result.values()):
        parser.error("The HC2 test is undefined for this input")
    args.output.write_text(json.dumps(result, allow_nan=False) + "\n",
                           encoding="utf-8")


if __name__ == "__main__":
    main()
