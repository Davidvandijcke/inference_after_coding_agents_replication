#!/usr/bin/env python3
"""Score the site-adjusted ITT effect on employment at 12 months.

Usage: python score.py --data <path> --output <path>
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t


def estimate(data_path: Path) -> tuple[float, float, float]:
    with data_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"site", "treatment", "employed_m12"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
        rows = list(reader)

    if not rows:
        raise ValueError("The input dataset has no observations")

    sites = [row["site"] for row in rows]
    site_levels = sorted(set(sites))
    treatment = np.asarray([float(row["treatment"]) for row in rows])
    outcome = np.asarray([float(row["employed_m12"]) for row in rows])

    # OLS with an intercept, site indicators, and treatment.
    columns = [np.ones(len(rows))]
    columns.extend(np.asarray([site == level for site in sites], dtype=float)
                   for level in site_levels[1:])
    columns.append(treatment)
    design = np.column_stack(columns)
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError("Treatment and site indicators are not separately estimable")

    xtx_inverse = np.linalg.inv(design.T @ design)
    coefficients = xtx_inverse @ (design.T @ outcome)
    residuals = outcome - design @ coefficients
    leverage = np.sum((design @ xtx_inverse) * design, axis=1)
    if np.any(leverage >= 1):
        raise ValueError("HC2 variance is undefined for an observation with leverage 1")

    adjusted = residuals**2 / (1.0 - leverage)
    meat = design.T @ (adjusted[:, None] * design)
    covariance = xtx_inverse @ meat @ xtx_inverse
    estimate = float(coefficients[-1])
    std_error = math.sqrt(float(covariance[-1, -1]))
    degrees_freedom = len(rows) - design.shape[1]
    if std_error <= 0 or degrees_freedom <= 0:
        raise ValueError("Could not calculate a valid treatment test")
    statistic = estimate / std_error
    p_value = float(2.0 * student_t.sf(abs(statistic), degrees_freedom))
    return estimate, std_error, p_value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    estimate_value, std_error, p_value = estimate(args.data)
    result = {
        "estimate": estimate_value,
        "std_error": std_error,
        "p_value": p_value,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
