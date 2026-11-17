#!/usr/bin/env python3
"""Estimate the site-adjusted ITT effect on employment at 12 months.

Usage: python score.py --data <path> --output <path>
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import t


def score(data_path):
    with open(data_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("The input dataset has no rows")

    sites = sorted({row["site"] for row in rows})
    treatment = np.array([float(row["treatment"]) for row in rows])
    outcome = np.array([float(row["employed_m12"]) for row in rows])
    site_columns = [np.array([float(row["site"] == site) for row in rows]) for site in sites[1:]]
    design = np.column_stack([np.ones(len(rows)), treatment, *site_columns])

    coefficients, _, rank, _ = np.linalg.lstsq(design, outcome, rcond=None)
    if rank != design.shape[1]:
        raise ValueError("Treatment and site indicators do not identify a full-rank model")

    residuals = outcome - design @ coefficients
    xtx_inverse = np.linalg.inv(design.T @ design)
    leverage = np.sum((design @ xtx_inverse) * design, axis=1)
    weighted_x = design * (residuals**2 / (1.0 - leverage))[:, None]
    covariance = xtx_inverse @ design.T @ weighted_x @ xtx_inverse

    estimate = float(coefficients[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    degrees_freedom = len(rows) - design.shape[1]
    p_value = float(2.0 * t.sf(abs(estimate / std_error), degrees_freedom))
    return {"estimate": estimate, "std_error": std_error, "p_value": p_value}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="CSV dataset with the supplied columns")
    parser.add_argument("--output", required=True, help="Path for the JSON result")
    args = parser.parse_args()
    result = score(args.data)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
