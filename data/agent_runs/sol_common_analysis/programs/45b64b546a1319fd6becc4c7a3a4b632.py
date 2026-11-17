#!/usr/bin/env python3
"""Estimate the Pathways to Work effect on 12-month employment.

Usage: python score.py --data data.csv --output result.json
Requires Python 3, numpy, pandas, and statsmodels.

The sole model is OLS: employed_m12 ~ treatment + C(site) +
employed_baseline. Inference uses HC2 standard errors and a two-sided
Student t-test with the OLS residual degrees of freedom. Estimates and
standard errors are in probability units, not percentage points.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def score(data_path):
    columns = ["employed_m12", "treatment", "site", "employed_baseline"]
    data = pd.read_csv(data_path, usecols=columns)
    if data.empty or data.isna().any().any():
        raise ValueError("The analysis columns must contain complete observations.")
    for column in ["employed_m12", "treatment", "employed_baseline"]:
        if not data[column].isin([0, 1]).all():
            raise ValueError(f"{column} must contain only 0 and 1.")
    if data["treatment"].nunique() != 2:
        raise ValueError("Both randomized treatment groups must be present.")

    model = smf.ols(
        "employed_m12 ~ treatment + C(site) + employed_baseline",
        data=data,
        missing="raise",
    )
    design = model.exog
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError("The specified regression design must have full rank.")
    if design.shape[0] <= design.shape[1]:
        raise ValueError("Positive residual degrees of freedom are required.")
    result = model.fit(cov_type="HC2", use_t=True)
    output = {
        "estimate": float(result.params["treatment"]),
        "std_error": float(result.bse["treatment"]),
        "p_value": float(result.pvalues["treatment"]),
    }
    if not all(np.isfinite(value) for value in output.values()):
        raise ValueError("The specified regression produced nonfinite inference.")
    return output


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
