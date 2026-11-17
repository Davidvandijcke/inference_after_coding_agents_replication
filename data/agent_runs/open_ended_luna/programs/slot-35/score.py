#!/usr/bin/env python3
"""Score the selected exploratory Pathways to Work subgroup comparison.

The estimand is the difference between the program's six-month effect on
debt collections for participants with education below college and those
with some college or more. The model includes site indicators and uses
HC2 heteroskedasticity-robust standard errors with a two-sided t test.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t


OUTCOME = "debt_collections_m6"
REQUIRED = {"site", "treatment", "education", OUTCOME}


def read_data(path):
    with open(path, newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = REQUIRED - set(reader.fieldnames or ())
        if missing:
            raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))
        rows = list(reader)

    if not rows:
        raise ValueError("Input data has no observations")

    try:
        site = [row["site"] for row in rows]
        treatment = np.asarray([float(row["treatment"]) for row in rows])
        education = np.asarray([float(row["education"]) for row in rows])
        outcome = np.asarray([float(row[OUTCOME]) for row in rows])
    except (TypeError, ValueError) as exc:
        raise ValueError("Required analysis columns must be complete and numeric") from exc

    if not (np.isfinite(treatment).all() and np.isfinite(education).all()
            and np.isfinite(outcome).all()):
        raise ValueError("Required analysis columns must be complete and finite")
    if not np.isin(treatment, [0.0, 1.0]).all():
        raise ValueError("treatment must contain only 0 and 1")
    return site, treatment, education, outcome


def score(data_path):
    site, treatment, education, outcome = read_data(data_path)
    below_college = (education < 3).astype(float)

    # Site indicators use the first observed site as the reference category.
    sites = sorted(set(site))
    columns = [
        np.ones(len(site)),
        treatment,
        below_college,
        treatment * below_college,
    ]
    columns.extend(np.asarray([value == level for value in site], dtype=float)
                   for level in sites[1:])
    design = np.column_stack(columns)

    n, k = design.shape
    if n <= k or np.linalg.matrix_rank(design) < k:
        raise ValueError("Insufficient observations or rank-deficient design")

    xtx_inv = np.linalg.inv(design.T @ design)
    coefficients = xtx_inv @ design.T @ outcome
    residuals = outcome - design @ coefficients

    # HC2 sandwich covariance: each squared residual is divided by 1 - h_ii.
    leverage = np.sum((design @ xtx_inv) * design, axis=1)
    if np.any(leverage >= 1.0):
        raise ValueError("HC2 covariance is undefined for this design")
    adjusted_sq_residuals = residuals**2 / (1.0 - leverage)
    meat = design.T @ (design * adjusted_sq_residuals[:, None])
    covariance = xtx_inv @ meat @ xtx_inv

    interaction_index = 3
    estimate = float(coefficients[interaction_index])
    std_error = float(np.sqrt(covariance[interaction_index, interaction_index]))
    if not np.isfinite(std_error) or std_error <= 0:
        raise ValueError("Could not calculate a positive finite standard error")
    statistic = estimate / std_error
    p_value = float(2.0 * student_t.sf(abs(statistic), df=n - k))

    return {
        "estimate": estimate,
        "std_error": std_error,
        "p_value": p_value,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="CSV dataset path")
    parser.add_argument("--output", required=True, help="JSON output path")
    args = parser.parse_args()

    result = score(args.data)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
