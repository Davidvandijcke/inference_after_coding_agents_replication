#!/usr/bin/env python3
"""Score the site-adjusted intention-to-treat effect on 12-month employment."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import t


def score(data_path):
    with open(data_path, newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))

    required = {"site", "treatment", "employed_m12"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("Data must contain site, treatment, and employed_m12 columns")

    site = np.array([int(row["site"]) for row in rows])
    treatment = np.array([float(row["treatment"]) for row in rows])
    outcome = np.array([float(row["employed_m12"]) for row in rows])

    # OLS with an intercept, treatment, and site indicators (site 1 is reference).
    sites = sorted(set(site))
    if len(sites) < 2:
        raise ValueError("At least two sites are required")
    design = np.column_stack(
        [np.ones(len(rows)), treatment]
        + [(site == value).astype(float) for value in sites[1:]]
    )
    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("Treatment and site indicators do not identify the model")

    xtx_inv = np.linalg.inv(design.T @ design)
    coefficients = xtx_inv @ design.T @ outcome
    residuals = outcome - design @ coefficients

    # HC2 covariance: each squared residual is divided by (1 - leverage).
    leverage = np.sum((design @ xtx_inv) * design, axis=1)
    if np.any(leverage >= 1):
        raise ValueError("HC2 covariance is undefined for an observation with leverage 1")
    weighted_design = design * (residuals**2 / (1.0 - leverage))[:, None]
    covariance = xtx_inv @ design.T @ weighted_design @ xtx_inv
    estimate = float(coefficients[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    degrees_freedom = len(outcome) - design.shape[1]
    statistic = estimate / std_error
    p_value = float(2 * t.sf(abs(statistic), degrees_freedom))

    return {"estimate": estimate, "std_error": std_error, "p_value": p_value}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV with the evaluation columns")
    parser.add_argument("--output", required=True, help="Path for the JSON result")
    args = parser.parse_args()

    result = score(args.data)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as destination:
        json.dump(result, destination, indent=2)
        destination.write("\n")


if __name__ == "__main__":
    main()
