#!/usr/bin/env python3
"""Estimate the intention-to-treat effect on employment at 12 months.

Usage: python score.py --data data.csv --output result.json
Requires numpy, pandas, and statsmodels.

The sole analysis is OLS of employed_m12 on an intercept, treatment, and
site indicators. Inference uses HC2 standard errors and a two-sided t-test
with residual degrees of freedom. Estimates are on the probability scale.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def analyze(data_path):
    data = pd.read_csv(
        data_path, usecols=["site", "treatment", "employed_m12"]
    )
    if data.empty or data.isna().any().any():
        raise ValueError("The analysis requires complete, nonempty data.")
    for column in ["treatment", "employed_m12"]:
        if not data[column].isin([0, 1]).all():
            raise ValueError(f"{column} must contain only 0 and 1.")
    if data["treatment"].nunique() != 2:
        raise ValueError("Both treatment and control participants are required.")

    sites = pd.get_dummies(
        data["site"], prefix="site", drop_first=True, dtype=float
    )
    design = pd.concat(
        [
            pd.Series(1.0, index=data.index, name="intercept"),
            data[["treatment"]].astype(float),
            sites,
        ],
        axis=1,
    )
    if len(data) <= design.shape[1]:
        raise ValueError("Positive residual degrees of freedom are required.")
    if np.linalg.matrix_rank(design.to_numpy()) != design.shape[1]:
        raise ValueError("The treatment effect is not identified by this design.")

    result = sm.OLS(data["employed_m12"], design).fit(
        cov_type="HC2", use_t=True
    )
    output = {
        "estimate": float(result.params["treatment"]),
        "std_error": float(result.bse["treatment"]),
        "p_value": float(result.pvalues["treatment"]),
    }
    if not all(np.isfinite(value) for value in output.values()):
        raise ValueError("The data do not support finite HC2 inference.")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error("The output path must differ from the input data path.")
    result = analyze(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
