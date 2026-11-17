#!/usr/bin/env python3
"""Score the site-adjusted treatment effect on employment at 12 months."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t


def estimate(data: pd.DataFrame) -> dict[str, float]:
    """Fit OLS with site indicators and report the HC2 treatment test."""
    y = data["employed_m12"].to_numpy(dtype=float)
    treatment = data["treatment"].to_numpy(dtype=float)
    site_indicators = pd.get_dummies(data["site"], drop_first=True, dtype=float)
    x = np.column_stack(
        (np.ones(len(data)), treatment, site_indicators.to_numpy(dtype=float))
    )

    xtx_inv = np.linalg.inv(x.T @ x)
    coefficients = xtx_inv @ x.T @ y
    residuals = y - x @ coefficients
    leverage = np.einsum("ij,jk,ik->i", x, xtx_inv, x)
    weighted_x = x * (residuals**2 / (1.0 - leverage))[:, None]
    covariance = xtx_inv @ (x.T @ weighted_x) @ xtx_inv

    estimate = float(coefficients[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    degrees_of_freedom = len(data) - x.shape[1]
    statistic = estimate / std_error
    p_value = float(2.0 * t.sf(abs(statistic), degrees_of_freedom))
    return {
        "estimate": estimate,
        "std_error": std_error,
        "p_value": p_value,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="CSV file with evaluation data")
    parser.add_argument("--output", required=True, help="Path for the JSON result")
    args = parser.parse_args()

    result = estimate(pd.read_csv(args.data))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
