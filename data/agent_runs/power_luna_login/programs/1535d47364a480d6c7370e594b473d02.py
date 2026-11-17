#!/usr/bin/env python3
"""Score the site-adjusted 12-month employment ITT effect.

Usage: python score.py --data <path> --output <path>
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import t


def estimate(data_path: Path) -> dict[str, float]:
    with data_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"site", "treatment", "employed_m12"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
        rows = list(reader)

    if not rows:
        raise ValueError("The input dataset has no rows")

    sites = np.array([int(row["site"]) for row in rows])
    treatment = np.array([float(row["treatment"]) for row in rows])
    outcome = np.array([float(row["employed_m12"]) for row in rows])

    # OLS with an intercept, treatment assignment, and site indicators.
    site_values = np.unique(sites)
    site_terms = [(sites == value).astype(float) for value in site_values[1:]]
    design = np.column_stack([np.ones(len(rows)), treatment, *site_terms])
    xtx_inverse = np.linalg.inv(design.T @ design)
    coefficients = xtx_inverse @ design.T @ outcome
    residuals = outcome - design @ coefficients

    # HC2 sandwich covariance: each squared residual is divided by (1 - h_ii).
    leverage = np.sum((design @ xtx_inverse) * design, axis=1)
    meat = design.T @ ((residuals**2 / (1.0 - leverage))[:, None] * design)
    covariance = xtx_inverse @ meat @ xtx_inverse

    estimate_value = float(coefficients[1])
    standard_error = float(np.sqrt(covariance[1, 1]))
    degrees_of_freedom = len(rows) - design.shape[1]
    statistic = estimate_value / standard_error
    p_value = float(2.0 * t.sf(abs(statistic), degrees_of_freedom))

    return {
        "estimate": estimate_value,
        "std_error": standard_error,
        "p_value": p_value,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Input CSV dataset")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON path")
    args = parser.parse_args()

    result = estimate(args.data)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
