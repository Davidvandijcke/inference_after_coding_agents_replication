#!/usr/bin/env python3
"""Score the site-adjusted treatment effect on employment at 12 months."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t


def estimate(data_path: Path) -> dict[str, float]:
    data = pd.read_csv(data_path)
    required = {"site", "treatment", "employed_m12"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    if data[list(required)].isna().any().any():
        raise ValueError("Required analysis columns contain missing values")

    # OLS with an intercept, treatment indicator, and site fixed effects.
    site_dummies = pd.get_dummies(data["site"], drop_first=True, dtype=float).to_numpy()
    design = np.column_stack(
        [np.ones(len(data)), data["treatment"].to_numpy(dtype=float), site_dummies]
    )
    outcome = data["employed_m12"].to_numpy(dtype=float)
    n, k = design.shape
    if n <= k:
        raise ValueError("Not enough observations to estimate the model")

    bread = np.linalg.inv(design.T @ design)
    coefficients = bread @ design.T @ outcome
    residuals = outcome - design @ coefficients

    # HC2 sandwich covariance: each squared residual is divided by 1 - leverage.
    leverage = np.sum((design @ bread) * design, axis=1)
    meat = design.T @ ((residuals**2 / (1.0 - leverage))[:, None] * design)
    covariance = bread @ meat @ bread
    treatment_se = float(np.sqrt(covariance[1, 1]))
    treatment_estimate = float(coefficients[1])
    statistic = treatment_estimate / treatment_se
    p_value = float(2.0 * t.sf(abs(statistic), df=n - k))

    return {
        "estimate": treatment_estimate,
        "std_error": treatment_se,
        "p_value": p_value,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Input CSV path")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON path")
    args = parser.parse_args()

    result = estimate(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
