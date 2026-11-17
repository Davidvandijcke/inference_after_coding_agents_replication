#!/usr/bin/env python3
"""Score the site-adjusted 12-month employment effect in a Pathways dataset."""

import argparse
import csv
import json
import sys

import numpy as np
from scipy.stats import t as student_t


def estimate(data_path):
    required = ("site", "treatment", "employed_m12")
    with open(data_path, newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        missing = set(required) - set(reader.fieldnames or ())
        if missing:
            raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))
        rows = list(reader)

    if not rows:
        raise ValueError("The input dataset has no rows")

    try:
        sites = [row["site"] for row in rows]
        treatment = np.asarray([float(row["treatment"]) for row in rows])
        outcome = np.asarray([float(row["employed_m12"]) for row in rows])
    except (TypeError, ValueError) as exc:
        raise ValueError("Required columns must contain numeric, nonmissing values") from exc

    if not np.isfinite(treatment).all() or not np.isfinite(outcome).all():
        raise ValueError("Required columns must contain finite values")

    # Intercept, treatment, and site indicators; the first sorted site is reference.
    site_levels = sorted(set(sites))
    site_columns = np.column_stack(
        [[float(site == level) for site in sites] for level in site_levels[1:]]
    ) if len(site_levels) > 1 else np.empty((len(rows), 0))
    design = np.column_stack((np.ones(len(rows)), treatment, site_columns))

    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("Treatment and site indicators are not separately estimable")

    xtx_inverse = np.linalg.inv(design.T @ design)
    coefficients = xtx_inverse @ design.T @ outcome
    residuals = outcome - design @ coefficients
    leverage = np.sum((design @ xtx_inverse) * design, axis=1)
    if np.any(leverage >= 1):
        raise ValueError("HC2 variance is undefined for this design")

    weighted_design = design * (residuals**2 / (1 - leverage))[:, None]
    covariance = xtx_inverse @ (design.T @ weighted_design) @ xtx_inverse
    estimate = float(coefficients[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    degrees_freedom = len(rows) - design.shape[1]
    if degrees_freedom <= 0 or std_error <= 0:
        raise ValueError("Insufficient data to calculate the treatment test")
    p_value = float(2 * student_t.sf(abs(estimate / std_error), degrees_freedom))
    return {"estimate": estimate, "std_error": std_error, "p_value": p_value}


def main():
    parser = argparse.ArgumentParser(
        description="Estimate the site-adjusted treatment effect on employment at month 12."
    )
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    result = estimate(args.data)
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, csv.Error) as error:
        print(f"score.py: {error}", file=sys.stderr)
        raise SystemExit(2)
