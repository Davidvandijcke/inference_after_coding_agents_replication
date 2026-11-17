#!/usr/bin/env python3
"""Score the 12-month earnings effect among participants employed at baseline."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


def estimate(data_path):
    data = pd.read_csv(data_path)
    data = data.loc[data["employed_baseline"] == 1]

    y = data["earnings_m12"].to_numpy(dtype=float)
    treatment = data["treatment"].to_numpy(dtype=float)
    site = data["site"].to_numpy()
    sites = sorted(np.unique(site))
    columns = [np.ones(len(data)), treatment]
    columns.extend((site == level).astype(float) for level in sites[1:])
    design = np.column_stack(columns)

    xtx_inverse = np.linalg.inv(design.T @ design)
    coefficients = xtx_inverse @ design.T @ y
    residuals = y - design @ coefficients
    leverage = np.sum((design @ xtx_inverse) * design, axis=1)
    meat = design.T @ ((residuals**2 / (1.0 - leverage))[:, None] * design)
    covariance = xtx_inverse @ meat @ xtx_inverse

    estimate = float(coefficients[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    degrees_freedom = len(data) - design.shape[1]
    statistic = estimate / std_error
    p_value = float(2.0 * student_t.sf(abs(statistic), degrees_freedom))
    return {
        "estimate": estimate,
        "std_error": std_error,
        "p_value": p_value,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="CSV with the evaluation columns")
    parser.add_argument("--output", required=True, help="Path for the JSON result")
    args = parser.parse_args()

    result = estimate(args.data)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
