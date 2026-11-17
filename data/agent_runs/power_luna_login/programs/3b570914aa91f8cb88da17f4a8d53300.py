#!/usr/bin/env python3
"""Estimate the site-adjusted 12-month employment effect."""

import argparse
import csv
import json
import sys

import numpy as np
from scipy.stats import t


def estimate(data_path):
    # This fixed specification matches the evaluation convention: OLS of
    # 12-month employment on assignment and site indicators, with HC2 errors.
    with open(data_path, newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError("The input dataset has no observations")

    treatment = np.asarray([float(row["treatment"]) for row in rows])
    outcome = np.asarray([float(row["employed_m12"]) for row in rows])
    sites = [row["site"] for row in rows]
    site_levels = sorted(set(sites))
    site_columns = [np.asarray([float(site == level) for site in sites])
                    for level in site_levels[1:]]
    design = np.column_stack([np.ones(len(rows)), treatment, *site_columns])

    n, k = design.shape
    if n <= k:
        raise ValueError("The dataset has too few observations for the model")

    xtx_inverse = np.linalg.inv(design.T @ design)
    coefficients = xtx_inverse @ design.T @ outcome
    residuals = outcome - design @ coefficients
    leverage = np.sum((design @ xtx_inverse) * design, axis=1)
    meat = design.T @ ((residuals**2 / (1.0 - leverage))[:, None] * design)
    covariance = xtx_inverse @ meat @ xtx_inverse

    standard_error = float(np.sqrt(covariance[1, 1]))
    estimate = float(coefficients[1])
    p_value = float(2.0 * t.sf(abs(estimate / standard_error), df=n - k))
    return {
        "estimate": estimate,
        "std_error": standard_error,
        "p_value": p_value,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Estimate the site-adjusted treatment effect on 12-month employment."
    )
    parser.add_argument("--data", required=True, help="Input CSV dataset")
    parser.add_argument("--output", required=True, help="Path for JSON results")
    args = parser.parse_args()

    try:
        result = estimate(args.data)
        with open(args.output, "w", encoding="utf-8") as output:
            json.dump(result, output, indent=2)
            output.write("\n")
    except (OSError, KeyError, ValueError, np.linalg.LinAlgError) as error:
        print(f"score.py: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
