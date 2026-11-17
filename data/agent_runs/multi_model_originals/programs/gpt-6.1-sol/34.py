#!/usr/bin/env python3
"""Estimate the fixed, full-sample effect on 12-month food insecurity.

Usage: python score.py --data data.csv --output result.json
Dependencies: numpy, pandas, scipy, statsmodels.

The analysis is OLS of food_insecure_m12 on an intercept, treatment, and
categorical site indicators. Inference uses HC2 standard errors and a
two-sided Student t test with OLS residual degrees of freedom. Estimates
and standard errors are in probability units (multiply by 100 for percentage
points). This program performs no outcome, subgroup, or model search.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def analyze(data_path):
    data = pd.read_csv(data_path)
    required = ["site", "treatment", "food_insecure_m12"]
    missing = [name for name in required if name not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    values = data[required].apply(pd.to_numeric, errors="raise")
    if values.empty or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("Analysis columns must contain finite, observed values.")
    if not set(values.treatment.unique()) == {0, 1}:
        raise ValueError("Treatment must be binary, with both arms represented.")
    if not values.food_insecure_m12.isin([0, 1]).all():
        raise ValueError("food_insecure_m12 must be binary (0 or 1).")

    design = pd.concat(
        [
            pd.Series(1.0, index=values.index, name="const"),
            values[["treatment"]].astype(float),
            pd.get_dummies(values.site, prefix="site", drop_first=True, dtype=float),
        ],
        axis=1,
    )
    if len(values) <= design.shape[1]:
        raise ValueError("Insufficient residual degrees of freedom.")
    if np.linalg.matrix_rank(design.to_numpy()) < design.shape[1]:
        raise ValueError("Treatment is not identified after accounting for site.")

    fit = sm.OLS(values.food_insecure_m12.astype(float), design).fit(
        cov_type="HC2", use_t=True
    )
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("The design does not permit finite HC2 inference.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error("The output path must differ from the input data path.")
    result = analyze(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
