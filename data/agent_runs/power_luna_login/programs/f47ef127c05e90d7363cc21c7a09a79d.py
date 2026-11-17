#!/usr/bin/env python3
"""Score the prespecified exploratory finding: 12-month employment among
participants employed at baseline, adjusted for randomization site.

Usage: python score.py --data <path> --output <path>
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


def score(data_path):
    data = pd.read_csv(data_path)
    required = {"site", "treatment", "employed_baseline", "employed_m12"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))

    subgroup = data.loc[data["employed_baseline"] == 1].copy()
    if subgroup.empty:
        raise ValueError("No participants employed at baseline")

    y = subgroup["employed_m12"].to_numpy(dtype=float)
    treatment = subgroup["treatment"].to_numpy(dtype=float)
    site_dummies = pd.get_dummies(
        subgroup["site"], drop_first=True, dtype=float
    ).to_numpy()
    X = np.column_stack((np.ones(len(subgroup)), treatment, site_dummies))

    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    residuals = y - X @ beta
    leverage = np.sum((X @ xtx_inv) * X, axis=1)

    # HC2 sandwich covariance, as specified in CODEBOOK.md.
    weighted_x = X * (residuals**2 / (1.0 - leverage))[:, None]
    covariance = xtx_inv @ (X.T @ weighted_x) @ xtx_inv
    std_error = float(np.sqrt(covariance[1, 1]))
    estimate = float(beta[1])
    degrees_freedom = len(y) - X.shape[1]
    t_statistic = estimate / std_error
    p_value = float(2.0 * student_t.sf(abs(t_statistic), degrees_freedom))

    return {
        "estimate": estimate,
        "std_error": std_error,
        "p_value": p_value,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()
    result = score(args.data)
    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2)
        output_file.write("\n")


if __name__ == "__main__":
    main()
