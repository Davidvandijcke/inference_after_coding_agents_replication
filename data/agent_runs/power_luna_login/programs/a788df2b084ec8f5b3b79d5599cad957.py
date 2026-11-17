#!/usr/bin/env python3
"""Score the site-adjusted intention-to-treat effect on 12-month employment."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t


def estimate(data: pd.DataFrame) -> tuple[float, float, float]:
    """OLS of employed_m12 on treatment and site indicators, with HC2 SE."""
    required = {"site", "treatment", "employed_m12"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    d = data[["site", "treatment", "employed_m12"]].dropna()
    if len(d) < 5:
        raise ValueError("At least five complete observations are required")
    if d["treatment"].nunique() != 2:
        raise ValueError("Both treatment and control observations are required")

    # Include an intercept, treatment, and indicators for every site except one.
    site_indicators = pd.get_dummies(d["site"], drop_first=True, dtype=float).to_numpy()
    x = np.column_stack(
        [np.ones(len(d)), d["treatment"].to_numpy(dtype=float), site_indicators]
    )
    y = d["employed_m12"].to_numpy(dtype=float)
    n, k = x.shape
    if n <= k:
        raise ValueError("Insufficient observations for the site-adjusted regression")

    xtx_inv = np.linalg.inv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    residual = y - x @ beta
    leverage = np.einsum("ij,jk,ik->i", x, xtx_inv, x)
    meat = x.T @ ((residual**2 / (1.0 - leverage))[:, None] * x)
    covariance = xtx_inv @ meat @ xtx_inv
    estimate = float(beta[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    p_value = float(2 * t.sf(abs(estimate / std_error), df=n - k))
    return estimate, std_error, p_value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV file")
    parser.add_argument("--output", required=True, help="Output JSON file")
    args = parser.parse_args()

    result = estimate(pd.read_csv(args.data))
    output = {
        "estimate": result[0],
        "std_error": result[1],
        "p_value": result[2],
    }
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
