#!/usr/bin/env python3
"""Score the site-adjusted 12-month employment effect in a trial dataset."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


def estimate_effect(data: pd.DataFrame) -> tuple[float, float, float]:
    """OLS treatment coefficient, HC2 standard error, and two-sided t-test."""
    required = {"site", "treatment", "employed_m12"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    if data[list(required)].isna().any().any():
        raise ValueError("Required analysis columns must not contain missing values")

    sites = sorted(data["site"].unique())
    if len(sites) < 2:
        raise ValueError("At least two sites are required for site adjustment")

    treatment = data["treatment"].to_numpy(dtype=float)
    columns = [np.ones(len(data)), treatment]
    columns.extend((data["site"].to_numpy() == site).astype(float) for site in sites[1:])
    design = np.column_stack(columns)
    outcome = data["employed_m12"].to_numpy(dtype=float)

    xtx_inverse = np.linalg.inv(design.T @ design)
    coefficients = xtx_inverse @ design.T @ outcome
    residuals = outcome - design @ coefficients
    leverage = np.einsum("ij,jk,ik->i", design, xtx_inverse, design)
    meat = design.T @ ((residuals**2 / (1.0 - leverage))[:, None] * design)
    covariance = xtx_inverse @ meat @ xtx_inverse

    estimate = float(coefficients[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    degrees_freedom = len(data) - design.shape[1]
    if degrees_freedom <= 0 or std_error <= 0:
        raise ValueError("Insufficient residual degrees of freedom for inference")
    statistic = estimate / std_error
    p_value = float(2.0 * student_t.sf(abs(statistic), degrees_freedom))
    return estimate, std_error, p_value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV dataset")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    estimate, std_error, p_value = estimate_effect(data)
    result = {
        "estimate": estimate,
        "std_error": std_error,
        "p_value": p_value,
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
