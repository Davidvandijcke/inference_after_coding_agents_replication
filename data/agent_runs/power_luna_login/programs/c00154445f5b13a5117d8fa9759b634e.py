#!/usr/bin/env python3
"""Estimate the site-adjusted 12-month employment effect for Pathways to Work."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t


BASELINE = [
    "age",
    "female",
    "parent",
    "household_size",
    "employed_baseline",
    "earnings_baseline",
    "months_unemployed_baseline",
]
REQUIRED = ["treatment", "site", "education", "employed_m12", *BASELINE]


def estimate(data_path: Path) -> dict[str, float]:
    data = pd.read_csv(data_path)
    missing = [column for column in REQUIRED if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if data[REQUIRED].isna().any().any():
        raise ValueError("Required analysis columns contain missing values")

    # OLS with an intercept, treatment, site indicators, and baseline covariates.
    # Education is categorical; the first observed category is the reference.
    pieces = [
        np.ones((len(data), 1)),
        data[["treatment"]].to_numpy(dtype=float),
        pd.get_dummies(data["site"], drop_first=True, dtype=float).to_numpy(),
        data[BASELINE].to_numpy(dtype=float),
        pd.get_dummies(data["education"], drop_first=True, dtype=float).to_numpy(),
    ]
    design = np.column_stack(pieces)
    outcome = data["employed_m12"].to_numpy(dtype=float)
    n, k = design.shape
    if n <= k or np.linalg.matrix_rank(design) != k:
        raise ValueError("Analysis design must have more observations than full-rank columns")

    xtx_inverse = np.linalg.inv(design.T @ design)
    coefficients = xtx_inverse @ design.T @ outcome
    residuals = outcome - design @ coefficients
    leverage = np.einsum("ij,jk,ik->i", design, xtx_inverse, design)
    hc2_meat = design.T @ ((residuals**2 / (1.0 - leverage))[:, None] * design)
    covariance = xtx_inverse @ hc2_meat @ xtx_inverse

    estimate = float(coefficients[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    degrees_freedom = n - k
    p_value = float(2 * t.sf(abs(estimate / std_error), degrees_freedom))
    return {
        "estimate": estimate,
        "std_error": std_error,
        "p_value": p_value,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    result = estimate(args.data)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
