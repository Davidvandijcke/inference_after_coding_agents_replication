#!/usr/bin/env python3
"""Score the selected lead analysis for Pathways to Work.

Fits OLS for 12-month food insecurity on treatment, site indicators, and the
available baseline covariates. Reports the treatment coefficient, its HC2
standard error, and the two-sided t-test p-value.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t


BASELINE_COVARIATES = (
    "age",
    "female",
    "parent",
    "education",
    "household_size",
    "employed_baseline",
    "earnings_baseline",
    "months_unemployed_baseline",
)
OUTCOME = "food_insecure_m12"


def estimate(data_path: Path) -> dict[str, float]:
    with data_path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        required = {"site", "treatment", OUTCOME, *BASELINE_COVARIATES}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
        rows = list(reader)

    if not rows:
        raise ValueError("The input data has no observations")

    sites = sorted({row["site"] for row in rows})
    columns = [
        np.ones(len(rows)),
        np.asarray([float(row["treatment"]) for row in rows]),
    ]
    # Omit the first site as the reference level.
    columns.extend(
        np.asarray([float(row["site"] == site) for row in rows])
        for site in sites[1:]
    )
    columns.extend(
        np.asarray([float(row[name]) for row in rows])
        for name in BASELINE_COVARIATES
    )
    design = np.column_stack(columns)
    outcome = np.asarray([float(row[OUTCOME]) for row in rows])

    n, k = design.shape
    rank = np.linalg.matrix_rank(design)
    if rank != k or n <= k:
        raise ValueError("The regression design is rank deficient or has no residual degrees of freedom")

    bread = np.linalg.inv(design.T @ design)
    coefficients = bread @ design.T @ outcome
    residuals = outcome - design @ coefficients
    leverage = np.einsum("ij,jk,ik->i", design, bread, design)
    if np.any(leverage >= 1):
        raise ValueError("HC2 adjustment is undefined for an observation with leverage at least 1")

    hc2_weights = residuals**2 / (1 - leverage)
    meat = design.T @ (design * hc2_weights[:, None])
    covariance = bread @ meat @ bread
    estimate_value = float(coefficients[1])
    standard_error = float(np.sqrt(covariance[1, 1]))
    degrees_freedom = n - rank
    statistic = estimate_value / standard_error
    p_value = float(2 * student_t.sf(abs(statistic), degrees_freedom))

    return {
        "estimate": estimate_value,
        "std_error": standard_error,
        "p_value": p_value,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Input CSV path")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON path")
    args = parser.parse_args()

    result = estimate(args.data)
    with args.output.open("w", encoding="utf-8") as destination:
        json.dump(result, destination, indent=2)
        destination.write("\n")


if __name__ == "__main__":
    main()
