#!/usr/bin/env python3
"""Estimate the site-adjusted 12-month employment effect."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import t


def estimate(data_path):
    with open(data_path, newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))

    required = {"site", "treatment", "employed_m12"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("Input CSV must contain site, treatment, and employed_m12 columns")

    site_values = [row["site"] for row in rows]
    treatment = np.asarray([float(row["treatment"]) for row in rows])
    outcome = np.asarray([float(row["employed_m12"]) for row in rows])
    sites = sorted(set(site_values))
    # Intercept, treatment indicator, and site indicators (first site omitted).
    site_indicators = np.column_stack(
        [[float(value == site) for value in site_values] for site in sites[1:]]
    ) if len(sites) > 1 else np.empty((len(rows), 0))
    design = np.column_stack((np.ones(len(rows)), treatment, site_indicators))

    n, k = design.shape
    if n <= k or len(set(treatment)) != 2:
        raise ValueError("Need more observations than model parameters and both treatment arms")

    bread = np.linalg.inv(design.T @ design)
    coefficients = bread @ design.T @ outcome
    residuals = outcome - design @ coefficients
    leverage = np.sum((design @ bread) * design, axis=1)
    meat = design.T @ ((residuals**2 / (1.0 - leverage))[:, None] * design)
    covariance = bread @ meat @ bread
    estimate = float(coefficients[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    p_value = float(2 * t.sf(abs(estimate / std_error), df=n - k))
    return {"estimate": estimate, "std_error": std_error, "p_value": p_value}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    result = estimate(args.data)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
