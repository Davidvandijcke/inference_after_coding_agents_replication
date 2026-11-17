#!/usr/bin/env python3
"""Score the site-adjusted treatment effect on 12-month employment."""

import argparse
import csv
import json
import sys

import numpy as np
from scipy.stats import t as student_t


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="CSV dataset")
    parser.add_argument("--output", required=True, help="Path for JSON results")
    args = parser.parse_args()

    with open(args.data, newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"site", "treatment", "employed_m12"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))
        rows = list(reader)

    if not rows:
        raise ValueError("Input dataset has no observations")

    try:
        sites = [int(row["site"]) for row in rows]
        treatment = np.array([float(row["treatment"]) for row in rows])
        outcome = np.array([float(row["employed_m12"]) for row in rows])
    except (TypeError, ValueError) as exc:
        raise ValueError("site, treatment, and employed_m12 must be numeric") from exc

    site_levels = sorted(set(sites))
    # Site indicators are the fixed effects, with the first site as reference.
    site_columns = np.array([[float(site == level) for level in site_levels[1:]]
                             for site in sites])
    design = np.column_stack((np.ones(len(rows)), treatment, site_columns))

    xtx_inverse = np.linalg.inv(design.T @ design)
    coefficients = xtx_inverse @ design.T @ outcome
    residuals = outcome - design @ coefficients
    leverage = np.sum((design @ xtx_inverse) * design, axis=1)
    weighted_design = design * (residuals ** 2 / (1.0 - leverage))[:, None]
    covariance = xtx_inverse @ design.T @ weighted_design @ xtx_inverse

    estimate = float(coefficients[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    df_resid = len(rows) - design.shape[1]
    if df_resid <= 0 or not np.isfinite(std_error) or std_error <= 0:
        raise ValueError("Insufficient data to estimate the treatment effect")
    statistic = estimate / std_error
    p_value = float(2.0 * student_t.sf(abs(statistic), df_resid))

    result = {
        "estimate": estimate,
        "std_error": std_error,
        "p_value": p_value,
    }
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
