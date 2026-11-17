#!/usr/bin/env python3
"""Score the site-adjusted 12-month employment effect in a Pathways dataset."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import t


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV with the evaluation columns")
    parser.add_argument("--output", required=True, help="Path for the JSON result")
    args = parser.parse_args()

    with open(args.data, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"site", "treatment", "employed_m12"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            parser.error("input is missing required columns: " + ", ".join(sorted(missing)))
        rows = list(reader)

    if not rows:
        parser.error("input CSV has no observations")

    sites = np.array([row["site"] for row in rows])
    treatment = np.array([float(row["treatment"]) for row in rows])
    outcome = np.array([float(row["employed_m12"]) for row in rows])
    site_levels = sorted(set(sites.tolist()))
    # Intercept, treatment, and site indicators; the first site is the reference.
    design = np.column_stack(
        [np.ones(len(rows)), treatment]
        + [(sites == site).astype(float) for site in site_levels[1:]]
    )

    xtx_inv = np.linalg.inv(design.T @ design)
    coefficients = xtx_inv @ design.T @ outcome
    residuals = outcome - design @ coefficients
    leverage = np.einsum("ij,jk,ik->i", design, xtx_inv, design)
    meat = design.T @ ((residuals**2 / (1.0 - leverage))[:, None] * design)
    covariance = xtx_inv @ meat @ xtx_inv

    estimate = float(coefficients[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    degrees_freedom = len(rows) - design.shape[1]
    p_value = float(2 * t.sf(abs(estimate / std_error), degrees_freedom))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            {"estimate": estimate, "std_error": std_error, "p_value": p_value},
            f,
            indent=2,
        )
        f.write("\n")


if __name__ == "__main__":
    main()
