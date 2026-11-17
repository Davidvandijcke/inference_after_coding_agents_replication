#!/usr/bin/env python3
"""Score the site-adjusted 12-month employment effect in a Pathways dataset."""

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
        raise ValueError("The input dataset has no observations.")

    sites = sorted({row["site"] for row in rows})
    treatment = np.asarray([float(row["treatment"]) for row in rows])
    outcome = np.asarray([float(row["employed_m12"]) for row in rows])
    site_columns = [
        np.asarray([float(row["site"] == site) for row in rows])
        for site in sites[1:]
    ]
    design = np.column_stack(
        [np.ones(len(rows)), treatment, *site_columns]
    )

    xtx_inv = np.linalg.inv(design.T @ design)
    coefficients = xtx_inv @ design.T @ outcome
    residuals = outcome - design @ coefficients
    leverage = np.einsum("ij,jk,ik->i", design, xtx_inv, design)
    hc2_weights = residuals**2 / (1.0 - leverage)
    covariance = xtx_inv @ (design.T @ (design * hc2_weights[:, None])) @ xtx_inv

    estimate = float(coefficients[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    degrees_freedom = len(rows) - design.shape[1]
    statistic = estimate / std_error
    p_value = float(2.0 * student_t.sf(abs(statistic), degrees_freedom))
    return {
        "estimate": estimate,
        "std_error": std_error,
        "p_value": p_value,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Estimate the site-adjusted treatment effect on employment at 12 months."
    )
    parser.add_argument("--data", required=True, help="Input CSV dataset")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    result = score(args.data)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
