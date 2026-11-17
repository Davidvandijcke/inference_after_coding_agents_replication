#!/usr/bin/env python3
"""Estimate the site-adjusted 12-month employment effect for Pathways to Work."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t


def score(data_path: Path) -> dict[str, float]:
    data = pd.read_csv(data_path)
    required = {"site", "treatment", "employed_m12"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    if data[list(required)].isna().any().any():
        raise ValueError("Required analysis columns contain missing values")

    treatment = data["treatment"].to_numpy(dtype=float)
    outcome = data["employed_m12"].to_numpy(dtype=float)
    site_indicators = pd.get_dummies(
        data["site"].astype(str), drop_first=True, dtype=float
    ).to_numpy()
    design = np.column_stack(
        [np.ones(len(data)), treatment, site_indicators]
    )

    xtx_inverse = np.linalg.inv(design.T @ design)
    coefficients = xtx_inverse @ design.T @ outcome
    residuals = outcome - design @ coefficients
    leverage = np.sum((design @ xtx_inverse) * design, axis=1)

    # HC2 sandwich covariance: each squared residual is divided by 1 - h_ii.
    weighted_crossproduct = design.T @ (
        (residuals**2 / (1.0 - leverage))[:, None] * design
    )
    covariance = xtx_inverse @ weighted_crossproduct @ xtx_inverse

    estimate = float(coefficients[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    degrees_freedom = len(data) - design.shape[1]
    statistic = estimate / std_error
    p_value = float(2 * t.sf(abs(statistic), degrees_freedom))
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

    result = score(args.data)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
