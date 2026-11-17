#!/usr/bin/env python3
"""Score the prespecified lead analysis: site-adjusted 12-month employment."""

import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats


def score(data_path):
    data = pd.read_csv(data_path)
    site = pd.get_dummies(data["site"], drop_first=True, dtype=float).to_numpy()
    treatment = data["treatment"].to_numpy(dtype=float)
    outcome = data["employed_m12"].to_numpy(dtype=float)
    design = np.column_stack((np.ones(len(data)), treatment, site))

    bread = np.linalg.inv(design.T @ design)
    coefficients = bread @ design.T @ outcome
    residuals = outcome - design @ coefficients
    leverage = np.sum((design @ bread) * design, axis=1)
    hc2_scores = design * (residuals / np.sqrt(1.0 - leverage))[:, None]
    covariance = bread @ (hc2_scores.T @ hc2_scores) @ bread

    estimate = coefficients[1]
    std_error = np.sqrt(covariance[1, 1])
    degrees_freedom = len(data) - design.shape[1]
    statistic = estimate / std_error
    p_value = 2.0 * stats.t.sf(abs(statistic), degrees_freedom)
    return {
        "estimate": float(estimate),
        "std_error": float(std_error),
        "p_value": float(p_value),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="CSV data path")
    parser.add_argument("--output", required=True, help="JSON output path")
    args = parser.parse_args()
    result = score(args.data)
    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2)
        output_file.write("\n")


if __name__ == "__main__":
    main()
