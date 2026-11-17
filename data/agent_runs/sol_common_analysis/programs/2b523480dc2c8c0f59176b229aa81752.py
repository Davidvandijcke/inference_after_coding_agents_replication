#!/usr/bin/env python3
"""Estimate the full-sample intention-to-treat effect on 12-month employment.

Usage: python score.py --data data.csv --output result.json
Model: employed_m12 ~ intercept + treatment + categorical site indicators.
Inference: HC2 standard error and two-sided t test, with residual degrees of
freedom. The reported p-value is for this single test, without multiplicity
adjustment. No other outcomes, subgroups, or specifications are evaluated.
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

    data = pd.read_csv(args.data, usecols=["treatment", "site", "employed_m12"])
    if data.empty or data.isna().any().any():
        raise ValueError("The chosen analysis requires complete observations.")
    if set(data["treatment"].unique()) != {0, 1}:
        raise ValueError("Treatment must contain both 0 and 1 only.")
    if not data["employed_m12"].isin([0, 1]).all():
        raise ValueError("employed_m12 must contain only 0 and 1.")

    site_indicators = pd.get_dummies(
        data["site"], prefix="site", drop_first=True, dtype=float
    )
    design = pd.concat(
        [pd.Series(1.0, index=data.index, name="intercept"),
         data["treatment"].astype(float), site_indicators],
        axis=1,
    )
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
        raise ValueError("The selected regression did not yield finite inference.")
    args.output.write_text(json.dumps(result, allow_nan=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
