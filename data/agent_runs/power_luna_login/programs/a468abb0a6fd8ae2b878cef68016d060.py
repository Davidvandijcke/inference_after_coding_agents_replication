#!/usr/bin/env python3
"""Score the 12-month employment effect among participants unemployed at baseline."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t


def estimate(data_path: Path) -> dict[str, float]:
    with data_path.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if int(row["employed_baseline"]) == 0]

    if not rows:
        raise ValueError("No participants unemployed at baseline were found.")

    sites = [row["site"] for row in rows]
    site_levels = sorted(set(sites))
    treatment = np.array([float(row["treatment"]) for row in rows])
    outcome = np.array([float(row["employed_m12"]) for row in rows])

    # OLS design: intercept, treatment, and site indicators (first site omitted).
    columns = [np.ones(len(rows)), treatment]
    columns.extend(np.array([float(site == level) for site in sites]) for level in site_levels[1:])
    design = np.column_stack(columns)

    xtx_inv = np.linalg.inv(design.T @ design)
    coefficients = xtx_inv @ design.T @ outcome
    residuals = outcome - design @ coefficients
    leverage = np.einsum("ij,jk,ik->i", design, xtx_inv, design)

    # HC2 sandwich covariance: each squared residual is divided by (1 - h_ii).
    weighted_design = design * (residuals**2 / (1.0 - leverage))[:, None]
    covariance = xtx_inv @ design.T @ weighted_design @ xtx_inv
    estimate = float(coefficients[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    degrees_freedom = len(rows) - design.shape[1]
    statistic = estimate / std_error
    p_value = float(2.0 * student_t.sf(abs(statistic), degrees_freedom))
    return {"estimate": estimate, "std_error": std_error, "p_value": p_value}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Input CSV dataset")
    parser.add_argument("--output", required=True, type=Path, help="Path for JSON result")
    args = parser.parse_args()

    result = estimate(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
