#!/usr/bin/env python3
"""Score the 6-month employment effect among participants unemployed at baseline."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t


def score(data_path: Path) -> dict[str, float]:
    with data_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {
            "site",
            "treatment",
            "employed_baseline",
            "employed_m6",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")

        rows = [row for row in reader if int(row["employed_baseline"]) == 0]

    if not rows:
        raise ValueError("No participants unemployed at baseline were found.")

    sites = sorted({int(row["site"]) for row in rows})
    site_base = sites[0]
    design_rows = []
    outcomes = []
    for row in rows:
        site = int(row["site"])
        treatment = float(row["treatment"])
        # Intercept, randomized treatment, and site fixed effects.
        design_rows.append(
            [1.0, treatment] + [float(site == level) for level in sites if level != site_base]
        )
        outcomes.append(float(row["employed_m6"]))

    x = np.asarray(design_rows, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    n, k = x.shape
    rank = int(np.linalg.matrix_rank(x))
    if rank != k or n <= rank:
        raise ValueError("The site-adjusted treatment model is not estimable for this dataset.")

    xtx_inv = np.linalg.inv(x.T @ x)
    beta = xtx_inv @ (x.T @ y)
    residuals = y - x @ beta

    # HC2 sandwich covariance: each squared residual is divided by 1 - leverage.
    leverage = np.einsum("ij,jk,ik->i", x, xtx_inv, x)
    if np.any(leverage >= 1.0):
        raise ValueError("HC2 variance is undefined because an observation has leverage 1.")
    adjusted_squares = residuals**2 / (1.0 - leverage)
    meat = x.T @ (x * adjusted_squares[:, None])
    covariance = xtx_inv @ meat @ xtx_inv

    estimate = float(beta[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    degrees_freedom = n - rank
    p_value = float(2.0 * student_t.sf(abs(estimate / std_error), degrees_freedom))
    return {
        "estimate": estimate,
        "std_error": std_error,
        "p_value": p_value,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate the site-adjusted 6-month employment effect among those unemployed at baseline."
    )
    parser.add_argument("--data", required=True, type=Path, help="Input CSV dataset")
    parser.add_argument("--output", required=True, type=Path, help="Path for the JSON result")
    args = parser.parse_args()

    if args.data.resolve() == args.output.resolve():
        parser.error("--output must be a separate file from --data")

    result = score(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
