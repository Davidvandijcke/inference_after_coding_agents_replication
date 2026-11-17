#!/usr/bin/env python3
"""Score the site-adjusted treatment effect on 12-month employment."""

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import t


def main():
    parser = argparse.ArgumentParser(
        description="Estimate the site-adjusted treatment effect on employed_m12."
    )
    parser.add_argument("--data", required=True, help="Input CSV dataset")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = {"site", "treatment", "employed_m12"}
    missing = required.difference(data.columns)
    if missing:
        parser.error("data is missing required columns: " + ", ".join(sorted(missing)))
    if data.empty or data[list(required)].isna().any().any():
        parser.error("required columns must contain at least one complete observation")

    # Intercept, treatment assignment, and site indicators (site 1 is reference).
    site = pd.get_dummies(data["site"], drop_first=True, dtype=float).to_numpy()
    x = np.column_stack(
        (np.ones(len(data)), data["treatment"].to_numpy(dtype=float), site)
    )
    y = data["employed_m12"].to_numpy(dtype=float)
    n, k = x.shape
    if n <= k:
        parser.error("not enough observations to estimate the model")

    xtx_inv = np.linalg.inv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    residual = y - x @ beta
    leverage = np.sum((x @ xtx_inv) * x, axis=1)
    meat = x.T @ ((residual**2 / (1.0 - leverage))[:, None] * x)
    covariance = xtx_inv @ meat @ xtx_inv

    estimate = float(beta[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    p_value = float(2.0 * t.sf(abs(estimate / std_error), df=n - k))

    with open(args.output, "w", encoding="utf-8") as output:
        json.dump(
            {"estimate": estimate, "std_error": std_error, "p_value": p_value},
            output,
            indent=2,
        )
        output.write("\n")


if __name__ == "__main__":
    main()
