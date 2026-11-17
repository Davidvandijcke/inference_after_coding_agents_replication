#!/usr/bin/env python3
"""Score the site-adjusted intention-to-treat effect on 12-month employment."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t


OUTCOME = "employed_m12"


def estimate(data_path):
    data = pd.read_csv(data_path)
    required = {"site", "treatment", OUTCOME}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    y = pd.to_numeric(data[OUTCOME], errors="raise").to_numpy(dtype=float)
    treatment = pd.to_numeric(data["treatment"], errors="raise").to_numpy(dtype=float)
    site = data["site"]
    site_indicators = pd.get_dummies(site, drop_first=True, dtype=float).to_numpy()
    design = np.column_stack((np.ones(len(data)), treatment, site_indicators))

    n, k = design.shape
    if n <= k:
        raise ValueError("The dataset must have more observations than model parameters.")

    xtx_inv = np.linalg.inv(design.T @ design)
    coefficients = xtx_inv @ design.T @ y
    residuals = y - design @ coefficients
    leverage = np.sum((design @ xtx_inv) * design, axis=1)
    hc2_meat = design.T @ ((residuals**2 / (1.0 - leverage))[:, None] * design)
    covariance = xtx_inv @ hc2_meat @ xtx_inv

    estimate = float(coefficients[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    degrees_freedom = n - k
    statistic = estimate / std_error
    p_value = float(2.0 * t.sf(abs(statistic), degrees_freedom))
    return {
        "estimate": estimate,
        "std_error": std_error,
        "p_value": p_value,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV dataset")
    parser.add_argument("--output", required=True, help="Path for JSON results")
    args = parser.parse_args()

    result = estimate(args.data)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2)
        output_file.write("\n")


if __name__ == "__main__":
    main()
