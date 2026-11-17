#!/usr/bin/env python3
"""Run the fixed Pathways to Work exploratory finding; no analysis search.

Target: effect of assignment on distress_m12 among participants with
employed_baseline == 0. OLS includes an intercept and categorical site fixed
effects, with HC2 standard errors and a two-sided residual-df Student t test.

Requires numpy, pandas, and statsmodels. Example:
    python score.py --data data.csv --output chosen_result.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def analyze(data_path):
    required = ["employed_baseline", "treatment", "site", "distress_m12"]
    data = pd.read_csv(data_path, usecols=required)
    if data[required].isna().any().any():
        raise ValueError("Required analysis columns must have no missing values.")
    for column in ("employed_baseline", "treatment"):
        if not data[column].isin([0, 1]).all():
            raise ValueError(f"{column} must contain only 0 and 1.")

    subgroup = data.loc[data["employed_baseline"].eq(0)].copy()
    if subgroup.empty or subgroup["treatment"].nunique() != 2:
        raise ValueError("The baseline-unemployed subgroup needs both trial arms.")

    # Treat site as categorical, regardless of its numeric labels.
    site_indicators = pd.get_dummies(
        subgroup["site"], prefix="site", drop_first=True, dtype=float
    )
    design = sm.add_constant(
        pd.concat([subgroup[["treatment"]], site_indicators], axis=1),
        has_constant="add",
    ).astype(float)
    outcome = subgroup["distress_m12"].astype(float)
    if not np.isfinite(design.to_numpy()).all() or not np.isfinite(outcome).all():
        raise ValueError("Analysis inputs must be finite.")
    if len(subgroup) <= design.shape[1]:
        raise ValueError("Insufficient residual degrees of freedom.")
    if np.linalg.matrix_rank(design.to_numpy()) != design.shape[1]:
        raise ValueError("Treatment and site effects are not jointly identifiable.")

    fit = sm.OLS(outcome, design).fit(cov_type="HC2", use_t=True)
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("The specified HC2 analysis is undefined for these data.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error("--output must differ from --data.")
    result = analyze(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
