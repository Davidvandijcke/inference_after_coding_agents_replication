#!/usr/bin/env python3
"""Score one fixed, exploratory Pathways to Work analysis.

Chosen estimand: assignment effect on earnings in the month preceding the
12-month follow-up among participants aged strictly less than 35 at baseline.
OLS: earnings_m12 ~ intercept + treatment + site indicators. Inference uses
HC2 standard errors and a two-sided Student t test with n - rank(X) degrees
of freedom. The reported p-value is nominal, without search adjustment.

Requires numpy, pandas, scipy, and statsmodels. Example:
    python score.py --data data.csv --output result.json

This file is self-contained and does not import or rerun exploration scripts.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def score(data_path):
    data = pd.read_csv(data_path)
    required = ["age", "site", "treatment", "earnings_m12"]
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if data[required].isna().any().any():
        raise ValueError("Required columns must be fully observed.")
    for column in ["age", "treatment", "earnings_m12"]:
        data[column] = pd.to_numeric(data[column], errors="raise")
        if not np.isfinite(data[column].to_numpy(dtype=float)).all():
            raise ValueError(f"Column {column} must contain finite numbers.")
    if not data.treatment.isin([0, 1]).all():
        raise ValueError("Treatment must be coded 0 or 1.")

    subgroup = data.loc[data.age.lt(35)].copy()
    if subgroup.treatment.nunique() != 2:
        raise ValueError("The under-35 subgroup must contain both trial arms.")
    design = pd.concat(
        [
            pd.Series(1.0, index=subgroup.index, name="const"),
            subgroup.treatment.astype(float),
            pd.get_dummies(subgroup.site, prefix="site", drop_first=True,
                           dtype=float),
        ],
        axis=1,
    )
    matrix = design.to_numpy(dtype=float)
    rank = np.linalg.matrix_rank(matrix)
    if rank != matrix.shape[1] or len(subgroup) <= rank:
        raise ValueError("The chosen regression needs a full-rank design and residual degrees of freedom.")
    leverage = np.sum(matrix * np.linalg.pinv(matrix).T, axis=1)
    if np.any(leverage >= 1.0 - 1e-12):
        raise ValueError("HC2 inference is undefined for an observation with leverage one.")

    fit = sm.OLS(subgroup.earnings_m12.astype(float), design).fit(
        cov_type="HC2", use_t=True
    )
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("The chosen regression did not produce finite inference.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error("The output path must differ from the input data path.")
    result = score(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
