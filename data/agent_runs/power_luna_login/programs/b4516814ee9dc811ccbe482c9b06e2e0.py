#!/usr/bin/env python3
"""Score the prespecified selected analysis on a Pathways-style dataset.

Estimates the treatment effect on employment at month 12 using OLS with site
fixed effects and HC2 heteroskedasticity-robust standard errors.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t


def score(data_path: Path) -> dict[str, float]:
    with data_path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))

    if not rows:
        raise ValueError("The input dataset has no observations")
    required = {"treatment", "site", "employed_m12"}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    treatment = np.asarray([float(row["treatment"]) for row in rows])
    outcome = np.asarray([float(row["employed_m12"]) for row in rows])
    sites = [row["site"] for row in rows]
    levels = sorted(set(sites))
    # Intercept, treatment, and site indicators (first site is reference).
    columns = [np.ones(len(rows)), treatment]
    columns.extend(np.asarray([site == level for site in sites], dtype=float)
                   for level in levels[1:])
    design = np.column_stack(columns)

    n, rank = design.shape[0], np.linalg.matrix_rank(design)
    if rank != design.shape[1] or n <= rank:
        raise ValueError("The treatment and site design matrix is not estimable")

    bread = np.linalg.inv(design.T @ design)
    coefficients = bread @ design.T @ outcome
    residuals = outcome - design @ coefficients
    leverage = np.einsum("ij,jk,ik->i", design, bread, design)
    meat = design.T @ (design * (residuals**2 / (1.0 - leverage))[:, None])
    covariance = bread @ meat @ bread
    estimate = float(coefficients[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    degrees_freedom = n - rank
    statistic = estimate / std_error
    p_value = float(2.0 * student_t.sf(abs(statistic), degrees_freedom))
    return {
        "estimate": estimate,
        "std_error": std_error,
        "p_value": p_value,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = score(args.data)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
