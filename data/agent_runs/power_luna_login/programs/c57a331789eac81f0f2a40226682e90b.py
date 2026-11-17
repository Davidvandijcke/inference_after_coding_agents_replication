#!/usr/bin/env python3
"""Score the site-adjusted 12-month employment effect in a Pathways dataset.

The treatment coefficient comes from OLS of employed_m12 on treatment and
site indicators. Its standard error uses the HC2 heteroskedasticity-robust
sandwich estimator, and the two-sided p-value uses a t distribution with
residual degrees of freedom.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t


def score(data_path):
    with open(data_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("The input dataset has no participant rows")

    try:
        treatment = np.array([float(row["treatment"]) for row in rows])
        outcome = np.array([float(row["employed_m12"]) for row in rows])
        sites = [row["site"] for row in rows]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Input must contain numeric treatment and employed_m12, plus site") from exc

    site_levels = sorted(set(sites), key=lambda value: float(value))
    site_columns = [np.array([site == level for site in sites], dtype=float)
                    for level in site_levels[1:]]
    design = np.column_stack([np.ones(len(rows)), treatment, *site_columns])

    n, k = design.shape
    if n <= k:
        raise ValueError("The dataset needs more observations than regression coefficients")

    xtx_inverse = np.linalg.inv(design.T @ design)
    coefficients = xtx_inverse @ design.T @ outcome
    residuals = outcome - design @ coefficients
    leverage = np.einsum("ij,jk,ik->i", design, xtx_inverse, design)
    weighted_design = design * (residuals**2 / (1.0 - leverage))[:, None]
    covariance = xtx_inverse @ design.T @ weighted_design @ xtx_inverse

    estimate = float(coefficients[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    statistic = estimate / std_error
    p_value = float(2.0 * student_t.sf(abs(statistic), df=n - k))
    return {"estimate": estimate, "std_error": std_error, "p_value": p_value}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to the participant CSV")
    parser.add_argument("--output", required=True, help="Path for the JSON result")
    args = parser.parse_args()
    result = score(args.data)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
