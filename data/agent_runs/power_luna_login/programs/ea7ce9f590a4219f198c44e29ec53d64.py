#!/usr/bin/env python3
"""Score the site-adjusted treatment effect on employment at 12 months."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t


def estimate(data: pd.DataFrame) -> dict[str, float]:
    """OLS coefficient and HC2 inference for employed_m12 ~ treatment + site."""
    site_indicators = pd.get_dummies(data["site"], drop_first=True, dtype=float)
    design = np.column_stack(
        [
            np.ones(len(data)),
            data["treatment"].to_numpy(dtype=float),
            site_indicators.to_numpy(dtype=float),
        ]
    )
    outcome = data["employed_m12"].to_numpy(dtype=float)

    xtx_inv = np.linalg.inv(design.T @ design)
    coefficients = xtx_inv @ design.T @ outcome
    residuals = outcome - design @ coefficients
    leverage = np.einsum("ij,jk,ik->i", design, xtx_inv, design)

    # HC2 uses squared residuals divided by (1 - leverage).
    weighted_x = design * (residuals**2 / (1.0 - leverage))[:, None]
    covariance = xtx_inv @ design.T @ weighted_x @ xtx_inv
    std_error = float(np.sqrt(covariance[1, 1]))
    estimate_value = float(coefficients[1])
    degrees_freedom = len(data) - design.shape[1]
    statistic = estimate_value / std_error
    p_value = float(2.0 * t.sf(abs(statistic), degrees_freedom))

    return {
        "estimate": estimate_value,
        "std_error": std_error,
        "p_value": p_value,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Input CSV file")
    parser.add_argument("--output", required=True, help="Output JSON file")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = {"site", "treatment", "employed_m12"}
    missing = required.difference(data.columns)
    if missing:
        parser.error("input is missing required columns: " + ", ".join(sorted(missing)))

    result = estimate(data)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
