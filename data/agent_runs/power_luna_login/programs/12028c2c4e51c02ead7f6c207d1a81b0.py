#!/usr/bin/env python3
"""Score the preselected 12-month employment analysis for baseline-unemployed participants."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="CSV dataset with the evaluation columns")
    parser.add_argument("--output", required=True, help="Path for the JSON result")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = {"site", "treatment", "employed_baseline", "employed_m12"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    # The prespecified scoring analysis is the site-adjusted treatment effect
    # on 12-month employment among participants unemployed at baseline.
    sample = data.loc[data["employed_baseline"] == 0].copy()
    if sample.empty:
        raise ValueError("No participants were unemployed at baseline")

    sites = sorted(sample["site"].dropna().unique())
    if not sites:
        raise ValueError("No observed sites in the analysis subgroup")
    site_values = sample["site"].to_numpy()
    columns = [
        np.ones(len(sample), dtype=float),
        sample["treatment"].to_numpy(dtype=float),
    ]
    columns.extend((site_values == site).astype(float) for site in sites[1:])
    design = np.column_stack(columns)
    outcome = sample["employed_m12"].to_numpy(dtype=float)

    if not np.isfinite(design).all() or not np.isfinite(outcome).all():
        raise ValueError("Analysis columns contain missing or non-finite values")

    n, k = design.shape
    df_resid = n - k
    if df_resid <= 0 or np.linalg.matrix_rank(design) != k:
        raise ValueError("Analysis design is not full rank or has no residual degrees of freedom")

    xtx_inv = np.linalg.inv(design.T @ design)
    coefficients = xtx_inv @ design.T @ outcome
    residuals = outcome - design @ coefficients
    leverage = np.sum((design @ xtx_inv) * design, axis=1)
    meat = design.T @ ((residuals**2 / (1.0 - leverage))[:, None] * design)
    covariance = xtx_inv @ meat @ xtx_inv

    estimate = float(coefficients[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    statistic = estimate / std_error
    p_value = float(2.0 * t.sf(abs(statistic), df_resid))

    output = {
        "estimate": estimate,
        "std_error": std_error,
        "p_value": p_value,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
