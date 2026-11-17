#!/usr/bin/env python3
"""Score the 12-month employment effect using site-adjusted OLS and HC2 SEs."""

import argparse
import csv
import json
import sys

import numpy as np
from scipy.stats import t


def analyze(path):
    with open(path, newline="", encoding="utf-8-sig") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError("data file has no observations")

    required = {"site", "treatment", "employed_m12"}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError("missing required columns: " + ", ".join(sorted(missing)))

    site = np.asarray([float(row["site"]) for row in rows])
    treatment = np.asarray([float(row["treatment"]) for row in rows])
    outcome = np.asarray([float(row["employed_m12"]) for row in rows])
    site_values = sorted(set(site))
    x = np.column_stack(
        [np.ones(len(rows)), treatment]
        + [(site == value).astype(float) for value in site_values[1:]]
    )

    xtx_inv = np.linalg.inv(x.T @ x)
    coefficients = xtx_inv @ x.T @ outcome
    residuals = outcome - x @ coefficients
    leverage = np.sum((x @ xtx_inv) * x, axis=1)
    meat = x.T @ ((residuals**2 / (1.0 - leverage))[:, None] * x)
    covariance = xtx_inv @ meat @ xtx_inv

    estimate = float(coefficients[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    degrees_freedom = len(rows) - x.shape[1]
    statistic = estimate / std_error
    p_value = float(2.0 * t.sf(abs(statistic), degrees_freedom))
    return {
        "estimate": estimate,
        "std_error": std_error,
        "p_value": p_value,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="input CSV with evaluation columns")
    parser.add_argument("--output", required=True, help="path for JSON result")
    args = parser.parse_args()
    try:
        result = analyze(args.data)
        with open(args.output, "w", encoding="utf-8") as output:
            json.dump(result, output, indent=2)
            output.write("\n")
    except (OSError, ValueError, KeyError, np.linalg.LinAlgError) as error:
        print(f"score.py: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
