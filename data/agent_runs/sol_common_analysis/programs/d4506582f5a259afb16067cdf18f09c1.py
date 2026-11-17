#!/usr/bin/env python3
"""Estimate the fixed six-month food-insecurity intention-to-treat effect.

Usage: python score.py --data data.csv --output result.json
Dependencies: numpy, pandas, statsmodels (and their dependencies).

OLS: food_insecure_m6 ~ intercept + treatment + categorical site indicators.
Inference: HC2 standard error and two-sided Student t test with n - rank(X)
degrees of freedom. The reported p-value is nominal, without adjustment for
the exploratory outcome search in the original study. No other outcomes,
subgroups, model specifications, or tests are evaluated by this program.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def score(data_path):
    data = pd.read_csv(
        data_path, usecols=["site", "treatment", "food_insecure_m6"]
    )
    if data.empty or data.isna().any().any():
        raise ValueError("The analysis requires complete, nonempty data.")
    if not data["treatment"].isin([0, 1]).all():
        raise ValueError("treatment must contain only 0 and 1.")
    if data["treatment"].nunique() != 2:
        raise ValueError("Both treatment and control observations are required.")
    if not data["food_insecure_m6"].isin([0, 1]).all():
        raise ValueError("food_insecure_m6 must contain only 0 and 1.")

    site_indicators = pd.get_dummies(
        data["site"], prefix="site", drop_first=True, dtype=float
    )
    design = sm.add_constant(
        pd.concat([data[["treatment"]], site_indicators], axis=1),
        has_constant="add",
    ).astype(float)
    rank = np.linalg.matrix_rank(design.to_numpy())
    if rank != design.shape[1] or len(data) <= rank:
        raise ValueError("The regression requires full rank and residual degrees of freedom.")

    fit = sm.OLS(data["food_insecure_m6"].astype(float), design).fit(
        cov_type="HC2", use_t=True
    )
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("The regression did not produce finite inference statistics.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error("The output must be a separate file from the input dataset.")
    result = score(args.data)
    args.output.write_text(json.dumps(result, allow_nan=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
