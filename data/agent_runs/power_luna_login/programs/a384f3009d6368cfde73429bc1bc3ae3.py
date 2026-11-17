#!/usr/bin/env python3
"""Score the site-adjusted treatment effect on 12-month employment."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


def score(data_path: str) -> dict[str, float]:
    data = pd.read_csv(data_path)
    required = {"site", "treatment", "employed_m12"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    if data[list(required)].isna().any().any():
        raise ValueError("Required analysis columns contain missing values")

    site = data["site"].to_numpy()
    sites = np.sort(data["site"].unique())
    columns = [
        np.ones(len(data), dtype=float),
        data["treatment"].to_numpy(dtype=float),
    ]
    columns.extend((site == value).astype(float) for value in sites[1:])
    design = np.column_stack(columns)
    outcome = data["employed_m12"].to_numpy(dtype=float)

    if len(data) <= design.shape[1] or np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("Insufficient data or rank-deficient site-adjusted design")

    bread = np.linalg.inv(design.T @ design)
    coefficients = bread @ design.T @ outcome
    residuals = outcome - design @ coefficients
    leverage = np.sum((design @ bread) * design, axis=1)
    meat = design.T @ ((residuals**2 / (1.0 - leverage))[:, None] * design)
    covariance = bread @ meat @ bread

    estimate = float(coefficients[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    if not np.isfinite(std_error) or std_error <= 0:
        raise ValueError("Could not calculate a finite standard error")
    degrees_freedom = len(data) - design.shape[1]
    statistic = estimate / std_error
    p_value = float(2.0 * student_t.sf(abs(statistic), degrees_freedom))
    return {
        "estimate": estimate,
        "std_error": std_error,
        "p_value": p_value,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate the site-adjusted treatment effect on 12-month employment."
    )
    parser.add_argument("--data", required=True, help="Input CSV dataset")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    result = score(args.data)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
