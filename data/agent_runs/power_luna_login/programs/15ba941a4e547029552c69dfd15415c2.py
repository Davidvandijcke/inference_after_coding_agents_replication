#!/usr/bin/env python3
"""Site-adjusted HC2 test of the 12-month employment effect."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t


OUTCOME = "employed_m12"


def score(data_path):
    with open(data_path, newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"site", "treatment", OUTCOME}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
        rows = list(reader)

    if not rows:
        raise ValueError("The input dataset has no observations")

    site = np.asarray([row["site"] for row in rows])
    treatment = np.asarray([float(row["treatment"]) for row in rows])
    outcome = np.asarray([float(row[OUTCOME]) for row in rows])

    # Intercept, treatment, and site indicators (first sorted site is reference).
    site_levels = sorted(set(site.tolist()))
    columns = [np.ones(len(rows)), treatment]
    columns.extend((site == level).astype(float) for level in site_levels[1:])
    design = np.column_stack(columns)
    n, k = design.shape
    df = n - k
    if df <= 0 or np.unique(treatment).size != 2:
        raise ValueError("Need residual degrees of freedom and both treatment arms")

    xtx_inverse = np.linalg.inv(design.T @ design)
    coefficients = xtx_inverse @ design.T @ outcome
    residuals = outcome - design @ coefficients
    leverage = np.sum((design @ xtx_inverse) * design, axis=1)
    weighted_crossproduct = (design.T * (residuals**2 / (1.0 - leverage))) @ design
    covariance = xtx_inverse @ weighted_crossproduct @ xtx_inverse

    estimate = float(coefficients[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    statistic = estimate / std_error
    p_value = float(2.0 * student_t.sf(abs(statistic), df))
    return {
        "estimate": estimate,
        "std_error": std_error,
        "p_value": p_value,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()
    result = score(args.data)
    with Path(args.output).open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
