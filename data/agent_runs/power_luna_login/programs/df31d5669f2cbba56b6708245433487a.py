#!/usr/bin/env python3
"""Score the site-adjusted intention-to-treat effect on 12-month employment."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="CSV with the evaluation columns")
    parser.add_argument("--output", required=True, help="Path for the JSON result")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = {"site", "treatment", "employed_m12"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))
    if data[list(required)].isna().any().any():
        raise ValueError("Required analysis columns must be complete")

    # OLS with an intercept, treatment indicator, and site indicators.
    sites = sorted(data["site"].unique())
    if len(sites) < 2:
        raise ValueError("At least two sites are required")
    treatment = data["treatment"].to_numpy(dtype=float)
    if not np.isin(treatment, [0.0, 1.0]).all():
        raise ValueError("treatment must be coded 0/1")
    y = data["employed_m12"].to_numpy(dtype=float)
    if not np.isin(y, [0.0, 1.0]).all():
        raise ValueError("employed_m12 must be coded 0/1")

    site_values = data["site"].to_numpy()
    X = np.column_stack(
        [np.ones(len(data)), treatment]
        + [(site_values == site).astype(float) for site in sites[1:]]
    )
    n, k = X.shape
    if n <= k or np.linalg.matrix_rank(X) < k:
        raise ValueError("The design matrix must have full rank and positive residual degrees of freedom")

    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    residuals = y - X @ beta
    leverage = np.sum((X @ xtx_inv) * X, axis=1)
    # HC2 sandwich covariance: each squared residual is divided by 1 - h_ii.
    weighted_x = X * (residuals**2 / (1.0 - leverage))[:, None]
    covariance = xtx_inv @ (X.T @ weighted_x) @ xtx_inv
    estimate = float(beta[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    p_value = float(2.0 * t.sf(abs(estimate / std_error), df=n - k))

    result = {
        "estimate": estimate,
        "std_error": std_error,
        "p_value": p_value,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
