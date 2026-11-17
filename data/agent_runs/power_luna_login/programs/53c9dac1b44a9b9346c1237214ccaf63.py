#!/usr/bin/env python3
"""Score the prespecified Pathways to Work 12-month employment effect."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


def estimate_effect(data: pd.DataFrame) -> tuple[float, float, float]:
    """OLS of employed_m12 on treatment and site indicators; HC2 inference."""
    required = {"site", "treatment", "employed_m12"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    frame = data[["site", "treatment", "employed_m12"]].dropna()
    if len(frame) < 2:
        raise ValueError("At least two complete observations are required")

    site_dummies = pd.get_dummies(frame["site"], drop_first=True, dtype=float).to_numpy()
    treatment = frame["treatment"].to_numpy(dtype=float)
    outcome = frame["employed_m12"].to_numpy(dtype=float)
    design = np.column_stack((np.ones(len(frame)), treatment, site_dummies))

    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("The treatment and site design matrix is rank deficient")
    residual_df = len(frame) - design.shape[1]
    if residual_df <= 0:
        raise ValueError("No residual degrees of freedom")

    xtx_inverse = np.linalg.inv(design.T @ design)
    coefficients = xtx_inverse @ design.T @ outcome
    residuals = outcome - design @ coefficients
    leverage = np.einsum("ij,jk,ik->i", design, xtx_inverse, design)
    if np.any(1.0 - leverage <= 0):
        raise ValueError("HC2 variance is undefined for an observation with unit leverage")

    weighted_design = design * (residuals**2 / (1.0 - leverage))[:, None]
    covariance = xtx_inverse @ (design.T @ weighted_design) @ xtx_inverse
    estimate = float(coefficients[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    statistic = estimate / std_error
    p_value = float(2.0 * student_t.sf(abs(statistic), df=residual_df))
    return estimate, std_error, p_value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    estimate, std_error, p_value = estimate_effect(pd.read_csv(args.data))
    result = {
        "estimate": estimate,
        "std_error": std_error,
        "p_value": p_value,
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
