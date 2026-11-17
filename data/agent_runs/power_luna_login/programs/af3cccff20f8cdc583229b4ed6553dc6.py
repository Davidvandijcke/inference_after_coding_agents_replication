#!/usr/bin/env python3
"""Score the site-adjusted intent-to-treat effect on 12-month employment.

Usage: python score.py --data <path> --output <path>

Fits OLS with an intercept, treatment assignment, and site fixed effects;
reports the HC2 standard error and two-sided t-test for the treatment effect.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats


def score(data_path):
    with open(data_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"site", "treatment", "employed_m12"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
        rows = list(reader)

    if not rows:
        raise ValueError("The input dataset has no observations")

    sites = sorted({row["site"] for row in rows})
    treatment = np.asarray([float(row["treatment"]) for row in rows])
    outcome = np.asarray([float(row["employed_m12"]) for row in rows])

    # Site 1 (the first sorted category) is the reference level.
    site_columns = np.asarray(
        [[float(row["site"] == site) for site in sites[1:]] for row in rows]
    )
    if not sites[1:]:
        site_columns = np.empty((len(rows), 0))
    design = np.column_stack((np.ones(len(rows)), treatment, site_columns))

    xtx_inv = np.linalg.inv(design.T @ design)
    beta = xtx_inv @ design.T @ outcome
    residuals = outcome - design @ beta
    leverage = np.sum((design @ xtx_inv) * design, axis=1)
    meat = design.T @ ((residuals**2 / (1.0 - leverage))[:, None] * design)
    covariance = xtx_inv @ meat @ xtx_inv

    estimate = float(beta[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    degrees_freedom = len(rows) - design.shape[1]
    statistic = estimate / std_error
    p_value = float(2.0 * stats.t.sf(abs(statistic), degrees_freedom))
    return {
        "estimate": estimate,
        "std_error": std_error,
        "p_value": p_value,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV dataset")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    result = score(args.data)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
