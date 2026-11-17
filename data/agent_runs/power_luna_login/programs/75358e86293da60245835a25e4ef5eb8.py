#!/usr/bin/env python3
"""Score the site-adjusted treatment effect on employment at 12 months.

Usage: python score.py --data <path> --output <path>
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


def estimate_effect(data_path: Path) -> dict[str, float]:
    data = pd.read_csv(data_path)
    outcome = data["employed_m12"].to_numpy(dtype=float)
    treatment = data["treatment"].to_numpy(dtype=float)
    site = pd.get_dummies(data["site"], drop_first=True, dtype=float).to_numpy()

    # OLS with an intercept, treatment indicator, and site fixed effects.
    design = np.column_stack((np.ones(len(data)), treatment, site))
    xtx_inverse = np.linalg.inv(design.T @ design)
    coefficients = xtx_inverse @ design.T @ outcome
    residuals = outcome - design @ coefficients

    # HC2 sandwich covariance: each squared residual is divided by (1 - h_ii).
    leverage = np.einsum("ij,jk,ik->i", design, xtx_inverse, design)
    weighted_design = design * (residuals**2 / (1.0 - leverage))[:, None]
    meat = design.T @ weighted_design
    covariance = xtx_inverse @ meat @ xtx_inverse
    estimate = float(coefficients[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    degrees_freedom = len(data) - design.shape[1]
    p_value = float(2.0 * student_t.sf(abs(estimate / std_error), degrees_freedom))

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

    result = estimate_effect(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
