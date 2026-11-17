#!/usr/bin/env python3
"""Score the selected Pathways to Work analysis on a supplied dataset.

The selected analysis is the site-adjusted intention-to-treat effect on
12-month psychological distress (Kessler-6): OLS of distress_m12 on treatment
and site indicators, with an HC2 robust standard error and a two-sided t-test.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


def estimate_effect(data_path: Path) -> dict[str, float]:
    data = pd.read_csv(data_path)
    required = {"distress_m12", "treatment", "site"}
    missing_columns = required.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing_columns))}")
    if data[list(required)].isna().any().any():
        raise ValueError("The analysis columns must not contain missing values")

    outcome = pd.to_numeric(data["distress_m12"], errors="raise").to_numpy(dtype=float)
    treatment = pd.to_numeric(data["treatment"], errors="raise").to_numpy(dtype=float)
    site_values = data["site"].to_numpy()
    n = len(data)
    if n < 2 or not np.isfinite(outcome).all() or not np.isfinite(treatment).all():
        raise ValueError("The analysis requires at least two rows and finite numeric values")
    if not np.isin(treatment, [0.0, 1.0]).all():
        raise ValueError("treatment must contain only 0 and 1")

    # Intercept, treatment, and k-1 site indicators (reference site omitted).
    site_levels = np.unique(site_values)
    site_indicators = np.column_stack(
        [(site_values == level).astype(float) for level in site_levels[1:]]
    ) if len(site_levels) > 1 else np.empty((n, 0))
    design = np.column_stack((np.ones(n), treatment, site_indicators))
    rank = np.linalg.matrix_rank(design)
    if rank != design.shape[1]:
        raise ValueError("Treatment and site indicators do not identify the selected model")
    residual_df = n - rank
    if residual_df <= 0:
        raise ValueError("The model has no residual degrees of freedom")

    bread = np.linalg.inv(design.T @ design)
    coefficients = bread @ design.T @ outcome
    residuals = outcome - design @ coefficients
    leverage = np.einsum("ij,jk,ik->i", design, bread, design)
    if np.any(1.0 - leverage <= 0.0):
        raise ValueError("HC2 variance is undefined for an observation with unit leverage")

    hc2_weights = residuals**2 / (1.0 - leverage)
    meat = (design.T * hc2_weights) @ design
    covariance = bread @ meat @ bread
    estimate = float(coefficients[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    if not np.isfinite(std_error) or std_error <= 0.0:
        raise ValueError("Could not calculate a positive finite HC2 standard error")
    statistic = estimate / std_error
    p_value = float(2.0 * student_t.sf(abs(statistic), df=residual_df))
    return {
        "estimate": estimate,
        "std_error": std_error,
        "p_value": p_value,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate the site-adjusted program effect on distress at 12 months."
    )
    parser.add_argument("--data", required=True, type=Path, help="Input CSV path")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON path")
    args = parser.parse_args()

    result = estimate_effect(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2)
        output_file.write("\n")


if __name__ == "__main__":
    main()
