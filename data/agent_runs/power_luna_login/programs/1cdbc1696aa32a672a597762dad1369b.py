#!/usr/bin/env python3
"""Score the 12-month earnings effect among participants unemployed at baseline.

Fits earnings_m12 on treatment and site indicators in the baseline-unemployed
subgroup, using HC2 heteroskedasticity-robust standard errors.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t


def estimate(data_path):
    with open(data_path, newline="", encoding="utf-8") as data_file:
        rows = list(csv.DictReader(data_file))

    required = {"site", "treatment", "employed_baseline", "earnings_m12"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("Input data must contain the required columns and at least one row")

    subgroup = [row for row in rows if float(row["employed_baseline"]) == 0]
    if not subgroup:
        raise ValueError("No participants are unemployed at baseline")

    sites = sorted({float(row["site"]) for row in subgroup})
    # Intercept, site indicators (first observed site is reference), treatment.
    design = np.column_stack(
        [
            np.ones(len(subgroup)),
            *[
                np.array([float(row["site"]) == site for row in subgroup], dtype=float)
                for site in sites[1:]
            ],
            np.array([float(row["treatment"]) for row in subgroup], dtype=float),
        ]
    )
    outcome = np.array([float(row["earnings_m12"]) for row in subgroup])

    xtx_inverse = np.linalg.inv(design.T @ design)
    coefficients = xtx_inverse @ design.T @ outcome
    residuals = outcome - design @ coefficients
    leverage = np.sum((design @ xtx_inverse) * design, axis=1)
    hc2_meat = design.T @ ((residuals**2 / (1.0 - leverage))[:, None] * design)
    covariance = xtx_inverse @ hc2_meat @ xtx_inverse

    treatment_index = design.shape[1] - 1
    estimate = float(coefficients[treatment_index])
    std_error = math.sqrt(float(covariance[treatment_index, treatment_index]))
    degrees_freedom = len(subgroup) - design.shape[1]
    statistic = estimate / std_error
    p_value = float(2.0 * student_t.sf(abs(statistic), degrees_freedom))
    return {"estimate": estimate, "std_error": std_error, "p_value": p_value}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to input CSV")
    parser.add_argument("--output", required=True, help="Path to output JSON")
    args = parser.parse_args()

    result = estimate(args.data)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2)
        output_file.write("\n")


if __name__ == "__main__":
    main()
