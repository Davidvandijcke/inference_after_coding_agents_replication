#!/usr/bin/env python3
"""Estimate the full-sample effect on employment at 12 months.

Usage: python score.py --data data.csv --output result.json
Requires pandas, numpy, and statsmodels (including its scipy dependency).
The model is employed_m12 ~ treatment + site indicators, with an intercept,
HC2 standard errors, and a two-sided t-test using residual degrees of freedom.
No other outcomes, subgroups, or model specifications are analyzed.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def score(data_path):
    data = pd.read_csv(data_path)
    required = ["site", "treatment", "employed_m12"]
    if data.empty or data[required].isna().any().any():
        raise ValueError("The analysis requires nonempty, complete observations.")
    for column in ["treatment", "employed_m12"]:
        if not data[column].isin([0, 1]).all():
            raise ValueError(f"{column} must contain only 0 and 1.")

    sites = pd.get_dummies(data["site"], prefix="site", drop_first=True,
                           dtype=float)
    design = sm.add_constant(
        pd.concat([data[["treatment"]].astype(float), sites], axis=1),
        has_constant="add",
    )
    if len(data) <= design.shape[1] or np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("The treatment effect requires a full-rank model and residual degrees of freedom.")
    leverage = np.sum(np.asarray(design) * np.linalg.pinv(design).T, axis=1)
    if np.any(leverage >= 1 - 1e-12):
        raise ValueError("HC2 is undefined for observations with unit leverage.")

    fit = sm.OLS(data["employed_m12"].astype(float), design).fit(
        cov_type="HC2", use_t=True
    )
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("The requested coefficient or its inference is not finite.")
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
