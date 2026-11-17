#!/usr/bin/env python3
"""Estimate the 12-month employment intention-to-treat effect.

Usage: python score.py --data data.csv --output result.json
Dependencies: numpy, pandas, statsmodels.

Model: employed_m12 ~ intercept + treatment + categorical site indicators.
Inference: HC2 standard error and a two-sided Student t test with n - rank(X)
degrees of freedom. The p-value is for this single coefficient, unadjusted for
the exploratory outcome comparisons used to select it in the original study.
Estimates and standard errors are in probability units (0.01 = 1 percentage
point). No baseline adjustment, subgroup selection, or other outcome is used.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def score(data_path):
    data = pd.read_csv(
        data_path, usecols=["site", "treatment", "employed_m12"]
    )
    if data.empty or data.isna().any().any():
        raise ValueError("The analysis columns must contain complete observations.")
    for name in ("treatment", "employed_m12"):
        if not data[name].isin([0, 1]).all():
            raise ValueError(f"{name} must contain only 0 and 1.")

    sites = pd.get_dummies(
        data["site"], prefix="site", drop_first=True, dtype=float
    )
    design = pd.concat(
        [pd.Series(1.0, index=data.index, name="intercept"),
         data[["treatment"]].astype(float), sites],
        axis=1,
    )
    if len(data) <= design.shape[1]:
        raise ValueError("The model needs positive residual degrees of freedom.")
    if np.linalg.matrix_rank(design.to_numpy()) != design.shape[1]:
        raise ValueError("Treatment and site effects are not separately identifiable.")

    fit = sm.OLS(data["employed_m12"], design, missing="raise").fit(
        cov_type="HC2", use_t=True
    )
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("The selected model did not produce finite inference.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error("--output must differ from --data.")
    result = score(args.data)
    with args.output.open("w", encoding="utf-8") as output:
        json.dump(result, output, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
