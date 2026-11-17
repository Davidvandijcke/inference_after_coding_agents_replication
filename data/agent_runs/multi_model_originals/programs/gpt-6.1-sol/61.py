#!/usr/bin/env python3
"""Run the fixed Pathways to Work analysis: food insecurity at 12 months.

Usage: python score.py --data data.csv --output result.json
Requires pandas and statsmodels. No outcome or subgroup search is performed.
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
    missing = set(required).difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    data = data[required].copy()
    if data.empty or data.isna().any().any():
        raise ValueError("Analysis columns must contain complete observations.")
    for column in ["treatment", "food_insecure_m12"]:
        data[column] = pd.to_numeric(data[column], errors="raise")
        if not data[column].isin([0, 1]).all():
            raise ValueError(f"{column} must contain only 0 and 1.")
    if data.treatment.nunique() != 2:
        raise ValueError("Both randomized treatment arms are required.")

    sites = pd.get_dummies(
        data["site"].astype("category"),
        prefix="site",
        drop_first=True,
        dtype=float,
    )
    design = sm.add_constant(
        pd.concat([data[["treatment"]], sites], axis=1), has_constant="add"
    )
    if np.linalg.matrix_rank(design.to_numpy(dtype=float)) != design.shape[1]:
        raise ValueError("Treatment and site coefficients are not identified.")
    if len(data) <= design.shape[1]:
        raise ValueError("Positive residual degrees of freedom are required.")

    model = sm.OLS(data["food_insecure_m12"], design).fit(
        cov_type="HC2", use_t=True
    )
    return {
        "estimate": float(model.params["treatment"]),
        "std_error": float(model.bse["treatment"]),
        "p_value": float(model.pvalues["treatment"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    # Protect the input when a caller accidentally reuses its path for output.
    if args.data.resolve() == args.output.resolve():
        parser.error("The output path must differ from the input path.")
    result = analyze(args.data)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
