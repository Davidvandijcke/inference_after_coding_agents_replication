#!/usr/bin/env python3
"""Score the site-adjusted treatment effect on employment at month 12.

Usage: python score.py --data <path> --output <path>

Fits OLS with a treatment indicator and site fixed effects, then reports the
treatment coefficient, HC2 standard error, and two-sided t-test p-value.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


def estimate(data: pd.DataFrame) -> dict[str, float]:
    required = {"site", "treatment", "employed_m12"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    if data[list(required)].isna().any().any():
        raise ValueError("Required analysis columns contain missing values")

    sites = sorted(data["site"].unique())
    if len(sites) < 2:
        raise ValueError("At least two sites are required for site adjustment")

    # Intercept, treatment, and site indicators (first site is the reference).
    site_columns = [data["site"].eq(site).to_numpy(dtype=float) for site in sites[1:]]
    x = np.column_stack(
        [np.ones(len(data)), data["treatment"].to_numpy(dtype=float), *site_columns]
    )
    y = data["employed_m12"].to_numpy(dtype=float)
    n, k = x.shape
    if n <= k or np.linalg.matrix_rank(x) < k:
        raise ValueError("The treatment and site design matrix is not estimable")

    xtx_inverse = np.linalg.inv(x.T @ x)
    coefficients = xtx_inverse @ x.T @ y
    residuals = y - x @ coefficients

    # HC2 sandwich covariance: (X'X)^-1 X' diag(e_i^2 / (1-h_i)) X (X'X)^-1.
    leverage = np.sum((x @ xtx_inverse) * x, axis=1)
    meat = x.T @ ((residuals**2 / (1.0 - leverage))[:, None] * x)
    covariance = xtx_inverse @ meat @ xtx_inverse
    estimate_value = float(coefficients[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    statistic = estimate_value / std_error
    p_value = float(2.0 * student_t.sf(abs(statistic), df=n - k))

    return {
        "estimate": estimate_value,
        "std_error": std_error,
        "p_value": p_value,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    result = estimate(pd.read_csv(args.data))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
