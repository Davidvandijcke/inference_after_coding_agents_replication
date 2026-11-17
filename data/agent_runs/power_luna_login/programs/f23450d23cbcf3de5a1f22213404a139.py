#!/usr/bin/env python3
"""Score the 12-month employment ITT effect with site fixed effects and HC2 SEs."""

import argparse
import csv
import json
import sys

import numpy as np
from scipy.stats import t as student_t


def estimate(data_path):
    with open(data_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    required = {"site", "treatment", "employed_m12"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("Data must include site, treatment, and employed_m12 columns")

    sites = [row["site"] for row in rows]
    treatment = np.array([float(row["treatment"]) for row in rows])
    outcome = np.array([float(row["employed_m12"]) for row in rows])
    site_values = sorted(set(sites))
    site_columns = np.column_stack(
        [[float(site == value) for site in sites] for value in site_values[1:]]
    ) if len(site_values) > 1 else np.empty((len(rows), 0))
    design = np.column_stack((np.ones(len(rows)), treatment, site_columns))

    xtx_inverse = np.linalg.inv(design.T @ design)
    coefficients = xtx_inverse @ design.T @ outcome
    residuals = outcome - design @ coefficients
    leverage = np.sum((design @ xtx_inverse) * design, axis=1)
    weighted_design = design * (residuals**2 / (1.0 - leverage))[:, None]
    covariance = xtx_inverse @ design.T @ weighted_design @ xtx_inverse

    estimate = float(coefficients[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    degrees_freedom = len(rows) - design.shape[1]
    p_value = float(2.0 * student_t.sf(abs(estimate / std_error), degrees_freedom))
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
    result = estimate(args.data)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, np.linalg.LinAlgError) as error:
        print(f"score.py: {error}", file=sys.stderr)
        sys.exit(2)
