#!/usr/bin/env python3
"""Score the 12-month employment effect among participants unemployed at baseline."""

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="CSV with the evaluation columns")
    parser.add_argument("--output", required=True, help="Path for the JSON result")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    needed = {"site", "treatment", "employed_baseline", "employed_m12"}
    missing = sorted(needed.difference(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    sample = data.loc[data["employed_baseline"] == 0].copy()
    if sample.empty:
        raise ValueError("No participants were unemployed at baseline")

    # Intercept, assignment indicator, and site indicators (first site omitted).
    sites = sorted(sample["site"].dropna().unique())
    if len(sites) < 2:
        raise ValueError("At least two sites are needed for site-adjusted estimation")
    columns = [np.ones(len(sample)), sample["treatment"].to_numpy(dtype=float)]
    columns.extend((sample["site"].to_numpy() == site).astype(float) for site in sites[1:])
    design = np.column_stack(columns)
    outcome = sample["employed_m12"].to_numpy(dtype=float)

    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("The site-adjusted design matrix is rank deficient")

    xtx_inv = np.linalg.inv(design.T @ design)
    coefficients = xtx_inv @ design.T @ outcome
    residuals = outcome - design @ coefficients
    leverage = np.einsum("ij,jk,ik->i", design, xtx_inv, design)
    meat = design.T @ ((residuals**2 / (1.0 - leverage))[:, None] * design)
    covariance = xtx_inv @ meat @ xtx_inv

    estimate = float(coefficients[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    df = len(sample) - design.shape[1]
    t_stat = estimate / std_error
    p_value = float(2.0 * student_t.sf(abs(t_stat), df))

    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(
            {"estimate": estimate, "std_error": std_error, "p_value": p_value},
            output_file,
            indent=2,
        )
        output_file.write("\n")


if __name__ == "__main__":
    main()
