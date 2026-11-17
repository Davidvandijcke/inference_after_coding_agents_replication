#!/usr/bin/env python3
"""Estimate the full-sample effect on food insecurity at 12 months.

Model: food_insecure_m12 ~ treatment + categorical site, with an intercept.
Inference: HC2 standard error and a two-sided residual-df Student t test.
The returned p_value is nominal for this single frozen test, not adjusted for
the outcome screening performed when choosing this analysis.

Requires pandas, numpy, and statsmodels.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def analyze(data_path):
    data = pd.read_csv(data_path)
    required = ["food_insecure_m12", "treatment", "site"]
    missing = set(required).difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if data[required].isna().any().any():
        raise ValueError("Required analysis columns contain missing values.")
    if not data["treatment"].isin([0, 1]).all():
        raise ValueError("treatment must be coded 0 or 1.")
    if not data["food_insecure_m12"].isin([0, 1]).all():
        raise ValueError("food_insecure_m12 must be coded 0 or 1.")

    site_indicators = pd.get_dummies(
        data["site"], prefix="site", drop_first=True, dtype=float
    )
    design = sm.add_constant(
        pd.concat([data[["treatment"]].astype(float), site_indicators], axis=1),
        has_constant="add",
    )
    if len(data) <= design.shape[1]:
        raise ValueError("Insufficient observations for residual degrees of freedom.")
    if np.linalg.matrix_rank(design.to_numpy()) != design.shape[1]:
        raise ValueError("The treatment and site design matrix is not full rank.")

    fit = sm.OLS(data["food_insecure_m12"].astype(float), design).fit(
        cov_type="HC2", use_t=True
    )
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("The model did not produce finite estimates and inference.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()
    if Path(args.data).resolve() == Path(args.output).resolve():
        parser.error("The output path must differ from the input data path.")
    result = analyze(args.data)
    with open(args.output, "w", encoding="utf-8") as output:
        json.dump(result, output, allow_nan=False, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
