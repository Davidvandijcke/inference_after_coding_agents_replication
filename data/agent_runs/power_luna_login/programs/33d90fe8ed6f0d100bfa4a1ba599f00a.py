#!/usr/bin/env python3
"""Score the 12-month employment effect from a site-stratified trial."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import t


def estimate(data_path: Path) -> dict[str, float]:
    with data_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError("data file has no participant rows")

    sites = sorted({int(row["site"]) for row in rows})
    # OLS: employed at 12 months ~ treatment + site fixed effects.
    # The first observed site is the reference category.
    columns = [
        np.ones(len(rows), dtype=float),
        np.array([float(row["treatment"]) for row in rows]),
    ]
    columns.extend(
        np.array([float(int(row["site"]) == site) for row in rows])
        for site in sites[1:]
    )
    x = np.column_stack(columns)
    y = np.array([float(row["employed_m12"]) for row in rows])

    xtx_inv = np.linalg.inv(x.T @ x)
    beta = xtx_inv @ (x.T @ y)
    residuals = y - x @ beta
    leverage = np.einsum("ij,jk,ik->i", x, xtx_inv, x)

    # HC2 sandwich covariance: divide each squared residual by 1 - h_ii.
    weighted_x = x * (residuals**2 / (1.0 - leverage))[:, None]
    meat = x.T @ weighted_x
    covariance = xtx_inv @ meat @ xtx_inv
    std_error = float(np.sqrt(covariance[1, 1]))
    estimate = float(beta[1])
    degrees_freedom = len(rows) - x.shape[1]
    statistic = estimate / std_error
    p_value = float(2.0 * t.sf(abs(statistic), degrees_freedom))

    return {
        "estimate": estimate,
        "std_error": std_error,
        "p_value": p_value,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate the site-adjusted 12-month employment effect."
    )
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    result = estimate(args.data)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
