#!/usr/bin/env python3
"""Run the fixed Pathways to Work analysis, without exploratory searching.

Estimand: treatment assignment effect on health_m6 among participants whose
baseline education is 3 (some college) or 4 (bachelor's degree or higher).
Model: OLS with an intercept, treatment, and categorical site fixed effects.
Inference: HC2 standard error; two-sided Student t test with residual degrees
of freedom. The returned p-value is nominal, without a search correction.

Requires numpy, pandas, scipy, and statsmodels.
Usage: python score.py --data data.csv --output result.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def score(data_path):
    columns = ["education", "site", "treatment", "health_m6"]
    data = pd.read_csv(data_path, usecols=columns)
    if data[columns].isna().any().any():
        raise ValueError("The required columns must have no missing values.")
    if not data["education"].isin([1, 2, 3, 4]).all():
        raise ValueError("education must be coded 1, 2, 3, or 4.")
    if not data["treatment"].isin([0, 1]).all():
        raise ValueError("treatment must be coded 0 or 1.")

    sample = data.loc[data["education"].isin([3, 4])].copy()
    if sample.empty or sample["treatment"].nunique() != 2:
        raise ValueError("The fixed education subgroup needs both treatment arms.")
    outcome = sample["health_m6"].astype(float)
    if not np.isfinite(outcome).all():
        raise ValueError("health_m6 must be finite.")

    sites = pd.get_dummies(
        sample["site"].astype("category"),
        prefix="site", drop_first=True, dtype=float,
    )
    design = pd.concat([sample[["treatment"]], sites], axis=1)
    design = sm.add_constant(design, has_constant="add").astype(float)
    rank = np.linalg.matrix_rank(design.to_numpy())
    if rank != design.shape[1] or len(sample) <= rank:
        raise ValueError("The fixed regression needs a full-rank design and positive residual degrees of freedom.")

    model = sm.OLS(outcome, design).fit(cov_type="HC2", use_t=True)
    result = {
        "estimate": float(model.params["treatment"]),
        "std_error": float(model.bse["treatment"]),
        "p_value": float(model.pvalues["treatment"]),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("The fixed model cannot produce finite HC2 inference for these observations.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    # Prevent accidental replacement of an input dataset.
    if args.data.resolve() == args.output.resolve():
        parser.error("--output must differ from --data")
    try:
        result = score(args.data)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
