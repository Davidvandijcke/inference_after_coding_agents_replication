#!/usr/bin/env python3
"""Score the site-adjusted 12-month employment effect in a Pathways dataset."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


def estimate(data_path: str) -> dict[str, float]:
    data = pd.read_csv(data_path)
    required = {"site", "treatment", "employed_m12"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    # OLS with an intercept, treatment indicator, and site fixed effects.
    site_effects = pd.get_dummies(data["site"], drop_first=True, dtype=float)
    x = np.column_stack(
        [np.ones(len(data)), data["treatment"].to_numpy(dtype=float), site_effects.to_numpy()]
    )
    y = data["employed_m12"].to_numpy(dtype=float)
    xtx_inv = np.linalg.inv(x.T @ x)
    coefficients = xtx_inv @ x.T @ y
    residuals = y - x @ coefficients

    # HC2 sandwich variance: residual squares are divided by (1 - leverage).
    leverage = np.einsum("ij,jk,ik->i", x, xtx_inv, x)
    adjusted_squares = residuals**2 / (1.0 - leverage)
    meat = x.T @ (x * adjusted_squares[:, None])
    covariance = xtx_inv @ meat @ xtx_inv
    estimate = float(coefficients[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    degrees_freedom = len(y) - np.linalg.matrix_rank(x)
    statistic = estimate / std_error
    p_value = float(2.0 * student_t.sf(abs(statistic), df=degrees_freedom))

    return {
        "estimate": estimate,
        "std_error": std_error,
        "p_value": p_value,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate the site-adjusted treatment effect on employment at 12 months."
    )
    parser.add_argument("--data", required=True, help="Input CSV dataset")
    parser.add_argument("--output", required=True, help="Path for JSON results")
    args = parser.parse_args()

    result = estimate(args.data)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
