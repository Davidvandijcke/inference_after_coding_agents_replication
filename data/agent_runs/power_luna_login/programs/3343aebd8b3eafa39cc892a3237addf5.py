#!/usr/bin/env python3
"""Score the 12-month employment effect in a Pathways to Work dataset."""

import argparse
import csv
import json
import sys

import numpy as np
from scipy.stats import t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="CSV dataset path")
    parser.add_argument("--output", required=True, help="JSON output path")
    args = parser.parse_args()

    with open(args.data, newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        required = {"site", "treatment", "employed_m12"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            parser.error("data must include site, treatment, and employed_m12 columns")
        rows = list(reader)

    if len(rows) < 5:
        parser.error("data must contain at least five observations")

    try:
        sites = np.array([int(row["site"]) for row in rows])
        treatment = np.array([float(row["treatment"]) for row in rows])
        outcome = np.array([float(row["employed_m12"]) for row in rows])
    except (TypeError, ValueError) as exc:
        parser.error(f"site, treatment, and employed_m12 must be numeric: {exc}")

    site_values = np.unique(sites)
    site_dummies = np.column_stack([sites == site for site in site_values[1:]]).astype(float)
    design = np.column_stack([np.ones(len(rows)), treatment, site_dummies])
    if np.linalg.matrix_rank(design) < design.shape[1]:
        parser.error("treatment and site indicators do not identify a full-rank model")

    xtx_inverse = np.linalg.inv(design.T @ design)
    coefficients = xtx_inverse @ design.T @ outcome
    residuals = outcome - design @ coefficients
    leverage = np.einsum("ij,jk,ik->i", design, xtx_inverse, design)
    weighted_crossproduct = design.T @ (
        (residuals**2 / (1.0 - leverage))[:, None] * design
    )
    covariance = xtx_inverse @ weighted_crossproduct @ xtx_inverse

    estimate = float(coefficients[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    degrees_freedom = len(rows) - design.shape[1]
    p_value = float(2.0 * t.sf(abs(estimate / std_error), degrees_freedom))

    result = {
        "estimate": estimate,
        "std_error": std_error,
        "p_value": p_value,
    }
    with open(args.output, "w", encoding="utf-8") as destination:
        json.dump(result, destination, indent=2)
        destination.write("\n")


if __name__ == "__main__":
    main()
