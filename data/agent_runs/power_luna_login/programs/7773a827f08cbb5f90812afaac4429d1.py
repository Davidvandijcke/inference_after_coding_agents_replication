#!/usr/bin/env python3
"""Score the site-adjusted intention-to-treat effect on 12-month employment."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t


OUTCOME = "employed_m12"


def estimate(data_path):
    with open(data_path, newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("The input dataset has no rows")

    required = {"site", "treatment", OUTCOME}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    sites = sorted({row["site"] for row in rows})
    treatment = np.asarray([float(row["treatment"]) for row in rows])
    outcome = np.asarray([float(row[OUTCOME]) for row in rows])
    site_values = np.asarray([row["site"] for row in rows])

    # Intercept, treatment assignment, and site indicators (first site omitted).
    site_indicators = np.column_stack(
        [(site_values == site).astype(float) for site in sites[1:]]
    ) if len(sites) > 1 else np.empty((len(rows), 0))
    design = np.column_stack([np.ones(len(rows)), treatment, site_indicators])
    n, k = design.shape
    if n <= k or np.linalg.matrix_rank(design) < k:
        raise ValueError("The site-adjusted regression design is not full rank")

    xtx_inverse = np.linalg.inv(design.T @ design)
    coefficients = xtx_inverse @ design.T @ outcome
    residuals = outcome - design @ coefficients
    leverage = np.sum((design @ xtx_inverse) * design, axis=1)
    weighted_crossproduct = design.T @ (
        (residuals**2 / (1.0 - leverage))[:, None] * design
    )
    covariance = xtx_inverse @ weighted_crossproduct @ xtx_inverse

    estimate = float(coefficients[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    statistic = estimate / std_error
    p_value = float(2.0 * student_t.sf(abs(statistic), df=n - k))
    return {
        "estimate": estimate,
        "std_error": std_error,
        "p_value": p_value,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV dataset")
    parser.add_argument("--output", required=True, help="Destination JSON file")
    args = parser.parse_args()

    result = estimate(args.data)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
