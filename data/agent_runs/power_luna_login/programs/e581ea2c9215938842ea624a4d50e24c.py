#!/usr/bin/env python3
"""Score the site-adjusted intention-to-treat effect on employment at 12 months."""

import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats


def score(data_path):
    data = pd.read_csv(data_path)
    sites = sorted(data["site"].dropna().unique().tolist())
    site_values = data["site"].to_numpy()
    site_columns = [
        (site_values == site).astype(float) for site in sites[1:]
    ]
    treatment = data["treatment"].to_numpy(dtype=float)
    outcome = data["employed_m12"].to_numpy(dtype=float)
    design = np.column_stack(
        [np.ones(len(data)), treatment, *site_columns]
    )

    bread = np.linalg.inv(design.T @ design)
    coefficients = bread @ design.T @ outcome
    residuals = outcome - design @ coefficients
    leverage = np.sum((design @ bread) * design, axis=1)
    hc2_meat = design.T @ (
        (residuals**2 / (1.0 - leverage))[:, None] * design
    )
    covariance = bread @ hc2_meat @ bread

    estimate = float(coefficients[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    residual_df = len(data) - design.shape[1]
    t_statistic = estimate / std_error
    p_value = float(2.0 * stats.t.sf(abs(t_statistic), residual_df))
    return {
        "estimate": estimate,
        "std_error": std_error,
        "p_value": p_value,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()
    result = score(args.data)
    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2)
        output_file.write("\n")


if __name__ == "__main__":
    main()
