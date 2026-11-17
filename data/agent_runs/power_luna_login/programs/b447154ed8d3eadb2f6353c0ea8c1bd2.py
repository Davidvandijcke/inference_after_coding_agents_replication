#!/usr/bin/env python3
"""Score the prespecified exploratory lead: 6-month distress among women.

Fits OLS within female participants, with treatment and site indicators, and
reports the treatment coefficient with an HC2 standard error and two-sided
Student t p-value.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats


def score(data_path):
    with open(data_path, newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    required = {"female", "treatment", "site", "distress_m6"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Input must contain columns: {', '.join(sorted(required))}")

    rows = [row for row in rows if int(row["female"]) == 1]
    if len(rows) < 6:
        raise ValueError("Too few female participants to fit the model")

    treatment = np.asarray([float(row["treatment"]) for row in rows])
    outcome = np.asarray([float(row["distress_m6"]) for row in rows])
    sites = [row["site"] for row in rows]
    site_levels = sorted(set(sites))
    site_columns = [np.asarray([site == level for site in sites], dtype=float)
                    for level in site_levels[1:]]
    design = np.column_stack(
        [np.ones(len(rows)), treatment, *site_columns]
    )
    n, k = design.shape
    if n <= k or np.linalg.matrix_rank(design) != k:
        raise ValueError("The treatment and site effects are not estimable")

    xtx_inv = np.linalg.inv(design.T @ design)
    coefficients = xtx_inv @ design.T @ outcome
    residuals = outcome - design @ coefficients
    leverage = np.sum((design @ xtx_inv) * design, axis=1)
    meat = design.T @ ((residuals**2 / (1.0 - leverage))[:, None] * design)
    covariance = xtx_inv @ meat @ xtx_inv

    estimate = float(coefficients[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    statistic = estimate / std_error
    p_value = float(2.0 * stats.t.sf(abs(statistic), df=n - k))
    return {"estimate": estimate, "std_error": std_error, "p_value": p_value}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="CSV input dataset")
    parser.add_argument("--output", required=True, help="Path for JSON result")
    args = parser.parse_args()

    result = score(args.data)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
