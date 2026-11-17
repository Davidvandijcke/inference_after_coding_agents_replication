#!/usr/bin/env python3
"""Estimate the full-sample effect of assignment on employment at 12 months.

Usage: python score.py --data data.csv --output result.json
Requires: numpy, pandas, scipy, statsmodels.
Model: employed_m12 ~ treatment + C(site), with an intercept.
Inference: HC2 standard error and a two-sided Student t-test using n - rank(X)
residual degrees of freedom. No baseline covariates or subgroup restrictions.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def score(data_path):
    data = pd.read_csv(
        data_path, usecols=["site", "treatment", "employed_m12"]
    )
    if data.empty or data.isna().any().any():
        raise ValueError("Required analysis columns must be nonempty and complete.")
    if set(data["treatment"].unique()) != {0, 1}:
        raise ValueError("Treatment must contain both randomized groups, 0 and 1.")
    if not data["employed_m12"].isin([0, 1]).all():
        raise ValueError("employed_m12 must contain only 0 and 1.")

    model = smf.ols("employed_m12 ~ treatment + C(site)", data=data)
    design = model.exog
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError("Treatment and site effects must be identifiable.")
    if design.shape[0] <= design.shape[1]:
        raise ValueError("Positive residual degrees of freedom are required.")
    fit = model.fit(cov_type="HC2", use_t=True)
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("The specified HC2 test is undefined for these data.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error("The output path must differ from the input data path.")
    result = score(args.data)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
