#!/usr/bin/env python3
"""Score the site-adjusted intention-to-treat effect on 12-month employment."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t


def estimate(data: pd.DataFrame) -> dict[str, float]:
    """OLS of employment_m12 on assignment and site indicators, with HC2 SE."""
    required = {"site", "treatment", "employed_m12"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    site_dummies = pd.get_dummies(data["site"], drop_first=True, dtype=float)
    x = np.column_stack(
        [np.ones(len(data)), data["treatment"].to_numpy(dtype=float), site_dummies.to_numpy()]
    )
    y = data["employed_m12"].to_numpy(dtype=float)

    xtx_inv = np.linalg.inv(x.T @ x)
    beta = xtx_inv @ (x.T @ y)
    residuals = y - x @ beta
    leverage = np.einsum("ij,jk,ik->i", x, xtx_inv, x)
    weighted_x = x * (residuals**2 / (1.0 - leverage))[:, None]
    meat = x.T @ weighted_x
    covariance = xtx_inv @ meat @ xtx_inv

    estimate_value = float(beta[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    degrees_freedom = len(data) - x.shape[1]
    p_value = float(2.0 * t.sf(abs(estimate_value / std_error), degrees_freedom))
    return {
        "estimate": estimate_value,
        "std_error": std_error,
        "p_value": p_value,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV file")
    parser.add_argument("--output", required=True, help="Output JSON file")
    args = parser.parse_args()

    result = estimate(pd.read_csv(args.data))
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
