#!/usr/bin/env python3
"""Score the selected lead analysis from the Pathways evaluation packet.

Fits employment at month 12 on randomized treatment assignment and site fixed
effects, with HC2 heteroskedasticity-robust standard errors.
"""

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


def score(data_path):
    data = pd.read_csv(data_path)
    required = {"site", "treatment", "employed_m12"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    if data[list(required)].isna().any().any():
        raise ValueError("Required analysis columns must not contain missing values")

    # Site indicators implement the trial's within-site randomization convention.
    site = pd.get_dummies(data["site"], drop_first=True, dtype=float).to_numpy()
    treatment = data["treatment"].to_numpy(dtype=float)
    outcome = data["employed_m12"].to_numpy(dtype=float)
    design = np.column_stack((np.ones(len(data)), treatment, site))

    if not np.isfinite(design).all() or not np.isfinite(outcome).all():
        raise ValueError("Analysis columns must contain finite numeric values")
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError("Treatment and site indicators do not identify the model")

    n, k = design.shape
    if n <= k:
        raise ValueError("Not enough observations to estimate the model")

    bread = np.linalg.inv(design.T @ design)
    coefficients = bread @ design.T @ outcome
    residuals = outcome - design @ coefficients
    leverage = np.sum((design @ bread) * design, axis=1)
    weighted_design = design * (residuals**2 / (1.0 - leverage))[:, None]
    covariance = bread @ design.T @ weighted_design @ bread

    estimate = float(coefficients[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    if not np.isfinite(std_error) or std_error <= 0:
        raise ValueError("Could not compute a positive finite standard error")
    p_value = float(2.0 * student_t.sf(abs(estimate / std_error), df=n - k))
    return {"estimate": estimate, "std_error": std_error, "p_value": p_value}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="CSV with the evaluation columns")
    parser.add_argument("--output", required=True, help="Path for the JSON result")
    args = parser.parse_args()
    result = score(args.data)
    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2)
        output_file.write("\n")


if __name__ == "__main__":
    main()
