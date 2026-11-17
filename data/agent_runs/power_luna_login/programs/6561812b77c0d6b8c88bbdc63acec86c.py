#!/usr/bin/env python3
"""Score the Pathways to Work 12-month employment treatment effect.

Usage: python score.py --data <path> --output <path>

Fits OLS of employed_m12 on treatment and site indicators and reports the
treatment coefficient, HC2 standard error, and two-sided t-test p-value.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t


def read_columns(path):
    with open(path, newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        required = {"site", "treatment", "employed_m12"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
        rows = list(reader)
    if not rows:
        raise ValueError("The input dataset has no rows")
    site = np.asarray([int(row["site"]) for row in rows], dtype=float)
    treatment = np.asarray([float(row["treatment"]) for row in rows])
    outcome = np.asarray([float(row["employed_m12"]) for row in rows])
    return site, treatment, outcome


def estimate(site, treatment, outcome):
    # Use the lowest observed site as the reference category.
    levels = np.unique(site)
    columns = [np.ones(len(site)), treatment]
    columns.extend((site == level).astype(float) for level in levels[1:])
    design = np.column_stack(columns)

    xtx_inverse = np.linalg.inv(design.T @ design)
    coefficients = xtx_inverse @ design.T @ outcome
    residuals = outcome - design @ coefficients
    leverage = np.sum((design @ xtx_inverse) * design, axis=1)

    # HC2 sandwich covariance: each squared residual is divided by 1 - h_ii.
    weighted_design = design * (residuals**2 / (1.0 - leverage))[:, None]
    meat = design.T @ weighted_design
    covariance = xtx_inverse @ meat @ xtx_inverse

    estimate = float(coefficients[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    df_resid = len(outcome) - design.shape[1]
    statistic = estimate / std_error
    p_value = float(2.0 * student_t.sf(abs(statistic), df_resid))
    return {"estimate": estimate, "std_error": std_error, "p_value": p_value}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV dataset")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    result = estimate(*read_columns(args.data))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
