#!/usr/bin/env python3
"""Estimate the full-sample ITT effect on employment at 12 months.

Frozen analysis: OLS of employed_m12 on an intercept, treatment, and site
indicators; HC2 standard error and a two-sided Student t test with residual
degrees of freedom. No subgroup selection, covariate selection, or search.

Dependencies: numpy, pandas, statsmodels (and its scipy dependency).
Usage: python score.py --data data.csv --output result.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def score(data_path):
    data = pd.read_csv(data_path)
    required = ["treatment", "site", "employed_m12"]
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    data = data[required]
    if data.empty or data.isna().any().any():
        raise ValueError("Analysis columns must be nonempty and fully observed.")
    for column in ("treatment", "employed_m12"):
        if not data[column].isin([0, 1]).all():
            raise ValueError(f"{column} must contain only 0 and 1.")
    if data.treatment.nunique() != 2:
        raise ValueError("Both treatment arms are required.")

    sites = pd.get_dummies(data.site, prefix="site", drop_first=True, dtype=float)
    design = sm.add_constant(
        pd.concat([data[["treatment"]].astype(float), sites], axis=1),
        has_constant="add",
    )
    if len(data) <= design.shape[1] or np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("The regression needs full rank and positive residual degrees of freedom.")
    fit = sm.OLS(data.employed_m12.astype(float), design).fit(cov_type="HC2", use_t=True)
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("The fitted model did not produce finite inference statistics.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = score(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
