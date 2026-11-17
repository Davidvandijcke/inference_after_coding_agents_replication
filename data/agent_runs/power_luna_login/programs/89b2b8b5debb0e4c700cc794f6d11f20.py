#!/usr/bin/env python3
"""Score the site-adjusted 12-month employment treatment effect."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t


def score(data_path: str) -> dict[str, float]:
    data = pd.read_csv(data_path)
    required = {"site", "treatment", "employed_m12"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    if data[list(required)].isna().any().any():
        raise ValueError("Required analysis columns contain missing values")

    # OLS design: intercept, treatment, and site indicators. The first
    # observed site is the reference category.
    sites = sorted(data["site"].unique())
    if len(sites) < 2:
        raise ValueError("At least two sites are required")
    site_values = data["site"].to_numpy()
    columns = [
        np.ones(len(data), dtype=float),
        data["treatment"].to_numpy(dtype=float),
    ]
    columns.extend((site_values == site).astype(float) for site in sites[1:])
    design = np.column_stack(columns)
    outcome = data["employed_m12"].to_numpy(dtype=float)

    n, k = design.shape
    df = n - k
    if df <= 0 or np.linalg.matrix_rank(design) < k:
        raise ValueError("The site-adjusted treatment model is not estimable")

    bread = np.linalg.inv(design.T @ design)
    coefficients = bread @ design.T @ outcome
    residuals = outcome - design @ coefficients
    leverage = np.sum((design @ bread) * design, axis=1)
    hc2_meat = design.T @ (
        (residuals**2 / (1.0 - leverage))[:, None] * design
    )
    covariance = bread @ hc2_meat @ bread

    estimate = float(coefficients[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    statistic = estimate / std_error
    p_value = float(2.0 * t.sf(abs(statistic), df))
    return {
        "estimate": estimate,
        "std_error": std_error,
        "p_value": p_value,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    result = score(args.data)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
