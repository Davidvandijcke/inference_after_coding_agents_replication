#!/usr/bin/env python3
"""Score the site-adjusted treatment effect on employment at 12 months."""

import argparse
import csv
import json
import sys

import numpy as np
from scipy.stats import t as student_t


def score(data_path):
    with open(data_path, newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("The input dataset has no observations")

    try:
        treatment = np.array([float(row["treatment"]) for row in rows])
        outcome = np.array([float(row["employed_m12"]) for row in rows])
        sites = [row["site"] for row in rows]
    except (KeyError, ValueError) as exc:
        raise ValueError("Input must contain numeric treatment and employed_m12, and site") from exc

    # Intercept, treatment, and site indicators (first observed site is reference).
    site_levels = list(dict.fromkeys(sites))
    columns = [np.ones(len(rows)), treatment]
    columns.extend(np.array([site == level for site in sites], dtype=float)
                   for level in site_levels[1:])
    design = np.column_stack(columns)
    n, rank = design.shape[0], np.linalg.matrix_rank(design)
    if rank != design.shape[1]:
        raise ValueError("Treatment and site indicators are not full rank")
    df_resid = n - rank
    if df_resid <= 0:
        raise ValueError("Insufficient residual degrees of freedom")

    xtx_inv = np.linalg.inv(design.T @ design)
    coefficients = xtx_inv @ design.T @ outcome
    residuals = outcome - design @ coefficients
    # HC2 uses each squared residual divided by one minus its leverage.
    leverage = np.einsum("ij,jk,ik->i", design, xtx_inv, design)
    if np.any(1.0 - leverage <= 0):
        raise ValueError("HC2 variance is undefined for an observation with unit leverage")
    weighted_crossprod = design.T @ (design * (residuals**2 / (1.0 - leverage))[:, None])
    covariance = xtx_inv @ weighted_crossprod @ xtx_inv

    estimate = float(coefficients[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    statistic = estimate / std_error
    p_value = float(2.0 * student_t.sf(abs(statistic), df_resid))
    return {"estimate": estimate, "std_error": std_error, "p_value": p_value}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="CSV dataset path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()
    try:
        result = score(args.data)
        with open(args.output, "w", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
    except Exception as exc:
        print(f"score.py: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
