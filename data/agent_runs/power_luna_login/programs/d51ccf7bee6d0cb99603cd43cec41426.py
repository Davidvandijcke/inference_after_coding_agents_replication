#!/usr/bin/env python3
"""Score the site-adjusted intention-to-treat effect on 12-month employment."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t


def estimate_effect(data_path):
    with Path(data_path).open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))

    required = {"treatment", "site", "employed_m12"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("Input must contain treatment, site, and employed_m12 columns")

    treatment = np.array([float(row["treatment"]) for row in rows])
    site = np.array([int(float(row["site"])) for row in rows])
    outcome = np.array([float(row["employed_m12"]) for row in rows])

    # OLS with an intercept, treatment, and site indicators (site 1 is reference).
    design = np.column_stack(
        [np.ones(len(rows)), treatment]
        + [(site == value).astype(float) for value in sorted(set(site))[1:]]
    )
    xtx_inverse = np.linalg.inv(design.T @ design)
    coefficients = xtx_inverse @ design.T @ outcome
    residuals = outcome - design @ coefficients

    # HC2 sandwich covariance: each squared residual is divided by 1 - leverage.
    leverage = np.sum((design @ xtx_inverse) * design, axis=1)
    weighted_design = design * (residuals**2 / (1.0 - leverage))[:, None]
    meat = design.T @ weighted_design
    covariance = xtx_inverse @ meat @ xtx_inverse
    std_error = float(np.sqrt(covariance[1, 1]))
    statistic = float(coefficients[1] / std_error)
    degrees_freedom = len(rows) - design.shape[1]
    p_value = float(2.0 * student_t.sf(abs(statistic), degrees_freedom))

    return {
        "estimate": float(coefficients[1]),
        "std_error": std_error,
        "p_value": p_value,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="CSV dataset path")
    parser.add_argument("--output", required=True, help="JSON output path")
    args = parser.parse_args()

    result = estimate_effect(args.data)
    with Path(args.output).open("w", encoding="utf-8") as destination:
        json.dump(result, destination, indent=2)
        destination.write("\n")


if __name__ == "__main__":
    main()
