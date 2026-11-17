#!/usr/bin/env python3
"""Run the single analysis chosen from the Pathways to Work exploration.

Estimand: effect of random assignment on six-month training enrollment among
participants with months_unemployed_baseline >= 12. The outcome is coded 0/1;
estimate and std_error are in probability units, not percentage points.

Specification: OLS training_m6 ~ treatment + categorical site indicators,
with an intercept, HC2 standard errors, and a two-sided residual-df t test.
No baseline adjustment, outcome selection, or subgroup search is performed.

Dependencies: numpy, pandas, scipy, statsmodels.
Usage: python score.py --data data.csv --output analysis/chosen_result.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def chosen_analysis(data):
    required = ["months_unemployed_baseline", "training_m6", "treatment", "site"]
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if data[required].isna().any().any():
        raise ValueError("Required analysis columns must have no missing values.")
    for col in ["months_unemployed_baseline", "training_m6", "treatment"]:
        data[col] = pd.to_numeric(data[col], errors="raise")
        if not np.isfinite(data[col]).all():
            raise ValueError(f"{col} must contain finite numeric values.")
    for col in ["training_m6", "treatment"]:
        if not data[col].isin([0, 1]).all():
            raise ValueError(f"{col} must be coded 0/1.")

    subset = data.loc[data["months_unemployed_baseline"] >= 12].copy()
    if subset.empty or subset["treatment"].nunique() != 2:
        raise ValueError("Chosen subgroup must contain both treatment arms.")

    site_dummies = pd.get_dummies(subset["site"].astype("category"),
                                  prefix="site", drop_first=True, dtype=float)
    design = pd.concat([subset[["treatment"]].astype(float), site_dummies], axis=1)
    design = sm.add_constant(design, has_constant="add")
    matrix = design.to_numpy()
    if len(subset) <= matrix.shape[1] or np.linalg.matrix_rank(matrix) < matrix.shape[1]:
        raise ValueError("Chosen regression must have full rank and residual degrees of freedom.")
    model = sm.OLS(subset["training_m6"].astype(float), design).fit(
        cov_type="HC2", use_t=True)
    result = {"estimate": float(model.params["treatment"]),
              "std_error": float(model.bse["treatment"]),
              "p_value": float(model.pvalues["treatment"])}
    if not all(np.isfinite(v) for v in result.values()):
        raise ValueError("The chosen regression did not produce finite inference.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error("Output must be separate from the input data.")
    result = chosen_analysis(pd.read_csv(args.data))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
