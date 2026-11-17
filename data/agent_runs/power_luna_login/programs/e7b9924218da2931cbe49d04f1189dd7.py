#!/usr/bin/env python3
"""Score the site-adjusted intention-to-treat effect on employment at month 12."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


def score(data_path: Path) -> dict[str, float]:
    data = pd.read_csv(data_path)
    required = {"site", "treatment", "employed_m12"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    if data[list(required)].isnull().any().any():
        raise ValueError("Required analysis columns contain missing values")

    treatment = data["treatment"].to_numpy(dtype=float)
    outcome = data["employed_m12"].to_numpy(dtype=float)
    sites = data["site"]
    site_values = sorted(sites.unique())
    # Intercept, treatment, and site indicators (first site is the reference).
    columns = [np.ones(len(data)), treatment]
    columns.extend((sites.to_numpy() == site).astype(float) for site in site_values[1:])
    design = np.column_stack(columns)

    xtx_inverse = np.linalg.inv(design.T @ design)
    coefficients = xtx_inverse @ design.T @ outcome
    residuals = outcome - design @ coefficients

    # HC2 sandwich covariance: each squared residual is divided by 1 - leverage.
    leverage = np.einsum("ij,jk,ik->i", design, xtx_inverse, design)
    adjusted_squares = residuals**2 / (1.0 - leverage)
    meat = design.T @ (adjusted_squares[:, None] * design)
    covariance = xtx_inverse @ meat @ xtx_inverse
    estimate = float(coefficients[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    degrees_freedom = len(data) - design.shape[1]
    statistic = estimate / std_error
    p_value = float(2.0 * student_t.sf(abs(statistic), degrees_freedom))
    return {"estimate": estimate, "std_error": std_error, "p_value": p_value}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Input CSV path")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON path")
    args = parser.parse_args()
    result = score(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
