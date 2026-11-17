#!/usr/bin/env python3
"""Score the site-adjusted treatment effect on employment at month 12."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t


OUTCOME = "employed_m12"


def estimate(data_path):
    with open(data_path, newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError("data file has no participant rows")

    treatments = np.array([float(row["treatment"]) for row in rows])
    outcomes = np.array([float(row[OUTCOME]) for row in rows])
    sites = [row["site"] for row in rows]
    site_levels = sorted(set(sites))
    # Intercept, treatment, and site indicators (first site is reference).
    columns = [np.ones(len(rows)), treatments]
    columns.extend(np.array([site == level for site in sites], dtype=float)
                   for level in site_levels[1:])
    design = np.column_stack(columns)

    n, k = design.shape
    if n <= k:
        raise ValueError("not enough observations to estimate the model")
    xtx_inverse = np.linalg.inv(design.T @ design)
    coefficients = xtx_inverse @ design.T @ outcomes
    residuals = outcomes - design @ coefficients
    leverage = np.sum((design @ xtx_inverse) * design, axis=1)
    adjusted_residual_sq = residuals**2 / (1.0 - leverage)
    meat = design.T @ (adjusted_residual_sq[:, None] * design)
    covariance = xtx_inverse @ meat @ xtx_inverse

    estimate = float(coefficients[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    statistic = estimate / std_error
    p_value = float(2.0 * student_t.sf(abs(statistic), df=n - k))
    return {"estimate": estimate, "std_error": std_error, "p_value": p_value}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="input CSV file")
    parser.add_argument("--output", required=True, help="output JSON file")
    args = parser.parse_args()
    result = estimate(args.data)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
