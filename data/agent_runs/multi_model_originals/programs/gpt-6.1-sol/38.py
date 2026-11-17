#!/usr/bin/env python3
"""Fixed analysis: six-month health among participants employed at baseline.

OLS health_m6 ~ intercept + treatment + site indicators, within
employed_baseline == 1. HC2 standard error; two-sided Student t test with
n - rank(X) degrees of freedom. No outcome or subgroup search is performed.

Dependencies: numpy, pandas, scipy.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t


def analyze(data):
    required = ["employed_baseline", "health_m6", "treatment", "site"]
    missing = set(required) - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if data[required].isna().any().any():
        raise ValueError("Required columns must have no missing values.")
    if not data["employed_baseline"].isin([0, 1]).all():
        raise ValueError("employed_baseline must be coded 0/1.")
    d = data.loc[data["employed_baseline"].eq(1), required]
    if set(d["treatment"].unique()) != {0, 1}:
        raise ValueError("The selected subgroup must contain both treatment arms.")
    site_dummies = pd.get_dummies(d["site"].astype(str), drop_first=True,
                                  dtype=float).to_numpy()
    X = np.column_stack((np.ones(len(d)), d["treatment"].to_numpy(float),
                         site_dummies))
    y = d["health_m6"].to_numpy(float)
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("The analysis requires finite numeric values.")
    n, k = X.shape
    if n <= k or np.linalg.matrix_rank(X) != k:
        raise ValueError("Insufficient observations or a rank-deficient design.")
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    bread = np.linalg.inv(X.T @ X)
    residual = y - X @ beta
    leverage = np.einsum("ij,jk,ik->i", X, bread, X)
    if np.any(leverage >= 1 - 1e-12):
        raise ValueError("HC2 is undefined for observations with leverage one.")
    # The treatment row of (X'X)^(-1) X' gives each observation's influence.
    influence = X @ bread[:, 1]
    variance = np.sum(influence**2 * residual**2 / (1 - leverage))
    se = float(np.sqrt(variance))
    estimate = float(beta[1])
    if not se > 0:
        raise ValueError("The treatment standard error must be positive.")
    p_value = float(2 * t.sf(abs(estimate / se), df=n - k))
    return {"estimate": estimate, "std_error": se, "p_value": p_value}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error("The output must be separate from the input data.")
    result = analyze(pd.read_csv(args.data))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
