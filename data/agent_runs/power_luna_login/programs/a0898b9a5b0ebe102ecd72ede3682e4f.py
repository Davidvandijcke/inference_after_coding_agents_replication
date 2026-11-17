#!/usr/bin/env python3
"""Score the 12-month employment treatment effect in a Pathways dataset."""

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
    if data[list(required)].isna().any().any():
        raise ValueError("Required analysis columns must not contain missing values")

    # OLS with an intercept, treatment indicator, and site fixed effects.
    sites = sorted(data["site"].unique())
    columns = [np.ones(len(data)), data["treatment"].to_numpy(dtype=float)]
    columns.extend((data["site"].to_numpy() == site).astype(float) for site in sites[1:])
    x = np.column_stack(columns)
    y = data["employed_m12"].to_numpy(dtype=float)

    xtx_inv = np.linalg.inv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    residuals = y - x @ beta

    # HC2 sandwich covariance: observation i is weighted by e_i^2 / (1 - h_i).
    leverage = np.sum((x @ xtx_inv) * x, axis=1)
    meat = x.T @ (x * (residuals**2 / (1.0 - leverage))[:, None])
    covariance = xtx_inv @ meat @ xtx_inv
    estimate = float(beta[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    degrees_freedom = len(y) - x.shape[1]
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
