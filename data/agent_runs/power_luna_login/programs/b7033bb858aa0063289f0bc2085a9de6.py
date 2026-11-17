#!/usr/bin/env python3
"""Score the site-adjusted treatment effect on 12-month employment."""

import argparse
import csv
import json
import sys

import numpy as np
from scipy.stats import t as student_t


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="CSV dataset")
    parser.add_argument("--output", required=True, help="JSON output path")
    args = parser.parse_args()

    with open(args.data, newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        required = {"site", "treatment", "employed_m12"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
        rows = list(reader)

    if not rows:
        raise ValueError("Dataset has no observations")

    sites = [row["site"] for row in rows]
    treatment = np.asarray([float(row["treatment"]) for row in rows])
    outcome = np.asarray([float(row["employed_m12"]) for row in rows])
    site_levels = sorted(set(sites))
    # Intercept, assignment indicator, and site indicators (first site omitted).
    design = np.column_stack(
        [np.ones(len(rows)), treatment]
        + [np.asarray([site == level for site in sites], dtype=float)
           for level in site_levels[1:]]
    )

    xtx_inv = np.linalg.inv(design.T @ design)
    coefficients = xtx_inv @ design.T @ outcome
    residuals = outcome - design @ coefficients
    leverage = np.sum((design @ xtx_inv) * design, axis=1)
    meat = design.T @ ((residuals**2 / (1.0 - leverage))[:, None] * design)
    covariance = xtx_inv @ meat @ xtx_inv

    estimate = float(coefficients[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    degrees_freedom = len(rows) - np.linalg.matrix_rank(design)
    statistic = estimate / std_error
    p_value = float(2.0 * student_t.sf(abs(statistic), degrees_freedom))

    with open(args.output, "w", encoding="utf-8") as destination:
        json.dump(
            {"estimate": estimate, "std_error": std_error, "p_value": p_value},
            destination,
            indent=2,
        )
        destination.write("\n")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, np.linalg.LinAlgError) as error:
        print(f"score.py: {error}", file=sys.stderr)
        sys.exit(2)
