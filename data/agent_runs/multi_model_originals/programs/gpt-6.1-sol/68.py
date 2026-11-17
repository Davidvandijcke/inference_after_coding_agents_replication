#!/usr/bin/env python3
"""Rerun the single selected Pathways to Work analysis.

Selection is fixed: participants with parent == 0; outcome hours_m12;
OLS with an intercept, treatment, and categorical site indicators; HC2
standard error; two-sided Student t test with n - rank(X) residual df.
There are no optional covariates, alternative outcomes, or model searches.

Dependencies: pandas, numpy, scipy, statsmodels.
Usage: python score.py --data data.csv --output analysis/score.json
The returned p_value is nominal (it does not adjust for the original search).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def chosen_analysis(data_path):
    columns = ["parent", "hours_m12", "treatment", "site"]
    data = pd.read_csv(data_path, usecols=columns)
    if data[columns].isna().any().any():
        raise ValueError("Required analysis columns contain missing values.")
    if not data["parent"].isin([0, 1]).all():
        raise ValueError("parent must be coded 0 or 1.")
    subset = data.loc[data["parent"].eq(0)].copy()
    if subset.empty or set(subset["treatment"].unique()) != {0, 1}:
        raise ValueError("The parent == 0 subgroup must contain both treatment arms.")
    design = pd.concat([
        subset[["treatment"]].astype(float),
        pd.get_dummies(subset["site"], prefix="site", drop_first=True, dtype=float),
    ], axis=1)
    design = sm.add_constant(design, has_constant="add")
    outcome = subset["hours_m12"].astype(float)
    matrix = design.to_numpy()
    if not np.isfinite(matrix).all() or not np.isfinite(outcome.to_numpy()).all():
        raise ValueError("The selected outcome and design must be finite.")
    if len(subset) <= matrix.shape[1] or np.linalg.matrix_rank(matrix) != matrix.shape[1]:
        raise ValueError("The selected regression needs full rank and positive residual df.")
    fit = sm.OLS(outcome, design).fit(cov_type="HC2", use_t=True)
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("The selected model produced non-finite inference.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error("--output must differ from --data.")
    result = chosen_analysis(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
