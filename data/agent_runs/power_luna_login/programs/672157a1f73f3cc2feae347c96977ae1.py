#!/usr/bin/env python3
"""Score the site-adjusted intention-to-treat effect on 12-month employment."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t


def estimate(data_path):
    with open(data_path, newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        required = {"site", "treatment", "employed_m12"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))
        rows = list(reader)

    if not rows:
        raise ValueError("The input dataset is empty")

    try:
        site = np.array([row["site"] for row in rows])
        treatment = np.array([float(row["treatment"]) for row in rows])
        outcome = np.array([float(row["employed_m12"]) for row in rows])
    except (TypeError, ValueError) as exc:
        raise ValueError("Required columns must contain numeric values") from exc

    sites = sorted(set(site.tolist()))
    # Intercept, randomized treatment, and site indicators (first site omitted).
    columns = [np.ones(len(rows)), treatment]
    columns.extend((site == level).astype(float) for level in sites[1:])
    design = np.column_stack(columns)
    n, k = design.shape
    if n <= k:
        raise ValueError("Not enough observations to estimate the model")

    xtx_inv = np.linalg.inv(design.T @ design)
    coefficients = xtx_inv @ design.T @ outcome
    residuals = outcome - design @ coefficients
    # HC2: scale each squared residual by 1 / (1 - leverage).
    leverage = np.einsum("ij,jk,ik->i", design, xtx_inv, design)
    if np.any(1.0 - leverage <= 0):
        raise ValueError("HC2 variance is undefined for this dataset")
    weighted_design = design * (residuals**2 / (1.0 - leverage))[:, None]
    covariance = xtx_inv @ (design.T @ weighted_design) @ xtx_inv
    estimate_value = float(coefficients[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    if not np.isfinite(std_error) or std_error <= 0:
        raise ValueError("Could not calculate a finite standard error")
    statistic = estimate_value / std_error
    p_value = float(2.0 * student_t.sf(abs(statistic), df=n - k))
    return {
        "estimate": estimate_value,
        "std_error": std_error,
        "p_value": p_value,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Input CSV dataset")
    parser.add_argument("--output", required=True, help="Path for JSON output")
    args = parser.parse_args()
    result = estimate(args.data)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
