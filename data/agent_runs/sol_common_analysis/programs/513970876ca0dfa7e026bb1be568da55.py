#!/usr/bin/env python3
"""Estimate the fixed 12-month earnings intention-to-treat effect.

Usage: python score.py --data data.csv --output result.json
Requires numpy, pandas, and statsmodels.

OLS: earnings_m12 ~ treatment + C(site) + earnings_baseline
                   + employed_baseline
Inference: HC2 standard error and a two-sided t-test with residual degrees
of freedom. No endpoint selection, subgroup search, or additional models.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def analyze(data_path):
    required = [
        "earnings_m12",
        "treatment",
        "site",
        "earnings_baseline",
        "employed_baseline",
    ]
    data = pd.read_csv(data_path, usecols=required)
    if data.empty or data.isna().any().any():
        raise ValueError("Analysis columns must contain complete observations.")
    numeric = [column for column in required if column != "site"]
    if not np.isfinite(data[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Analysis variables must be finite numeric values.")
    if set(data["treatment"].unique()) != {0, 1}:
        raise ValueError("Both treatment groups, coded 0 and 1, are required.")
    if not data["employed_baseline"].isin([0, 1]).all():
        raise ValueError("employed_baseline must be coded 0 or 1.")

    model = smf.ols(
        "earnings_m12 ~ treatment + C(site) + earnings_baseline"
        " + employed_baseline",
        data=data,
        missing="raise",
    )
    design = model.exog
    if (
        len(data) <= design.shape[1]
        or np.linalg.matrix_rank(design) != design.shape[1]
    ):
        raise ValueError("The specified regression needs a full-rank design and residual degrees of freedom.")
    result = model.fit(cov_type="HC2", use_t=True)
    output = {
        "estimate": float(result.params["treatment"]),
        "std_error": float(result.bse["treatment"]),
        "p_value": float(result.pvalues["treatment"]),
    }
    if not all(np.isfinite(value) for value in output.values()):
        raise ValueError("The specified HC2 inference is undefined for this dataset.")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error("--output must differ from --data")
    output = analyze(args.data)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
