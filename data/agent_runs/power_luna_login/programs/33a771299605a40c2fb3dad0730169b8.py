#!/usr/bin/env python3
"""Estimate the prespecified-for-scoring month-6 food insecurity contrast.

Fits OLS of food_insecure_m6 on treatment and site indicators, with an
intercept, and reports the treatment coefficient, HC2 standard error, and
two-sided t-test p-value. The script deliberately performs no model search.
"""

import argparse
import csv
import json
import sys

import numpy as np
from scipy.stats import t as student_t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    with open(args.data, newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        required = {"site", "treatment", "food_insecure_m6"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            parser.error("input CSV is missing required columns: " + ", ".join(sorted(missing)))
        rows = list(reader)

    if not rows:
        parser.error("input CSV has no observations")

    sites = [row["site"] for row in rows]
    site_levels = sorted(set(sites))
    treatment = np.asarray([float(row["treatment"]) for row in rows])
    outcome = np.asarray([float(row["food_insecure_m6"]) for row in rows])

    # Omit the first site indicator to avoid collinearity with the intercept.
    columns = [np.ones(len(rows)), treatment]
    columns.extend(np.asarray([site == level for site in sites], dtype=float)
                   for level in site_levels[1:])
    design = np.column_stack(columns)
    n, k = design.shape
    if n <= k or len(set(treatment)) < 2:
        parser.error("input must contain more observations than model parameters and both arms")

    bread = np.linalg.inv(design.T @ design)
    coefficients = bread @ design.T @ outcome
    residuals = outcome - design @ coefficients
    leverage = np.einsum("ij,jk,ik->i", design, bread, design)
    hc2_meat = design.T @ ((residuals ** 2 / (1.0 - leverage))[:, None] * design)
    covariance = bread @ hc2_meat @ bread

    estimate = float(coefficients[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    degrees_freedom = n - k
    statistic = estimate / std_error
    p_value = float(2.0 * student_t.sf(abs(statistic), degrees_freedom))

    with open(args.output, "w", encoding="utf-8") as destination:
        json.dump({"estimate": estimate, "std_error": std_error, "p_value": p_value}, destination)
        destination.write("\n")


if __name__ == "__main__":
    main()
