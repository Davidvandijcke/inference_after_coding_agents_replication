#!/usr/bin/env python3
"""Score the 12-month employment effect using the Pathways analysis convention."""

import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats


def estimate(data: pd.DataFrame) -> dict[str, float]:
    """OLS of employed_m12 on treatment and site indicators, with HC2 SE."""
    sites = sorted(data["site"].unique())
    columns = [np.ones(len(data)), data["treatment"].to_numpy(dtype=float)]
    columns.extend((data["site"].to_numpy() == site).astype(float) for site in sites[1:])
    design = np.column_stack(columns)
    outcome = data["employed_m12"].to_numpy(dtype=float)

    xtx_inverse = np.linalg.inv(design.T @ design)
    coefficients = xtx_inverse @ design.T @ outcome
    residuals = outcome - design @ coefficients
    leverage = np.sum((design @ xtx_inverse) * design, axis=1)
    weighted_x = design * (residuals**2 / (1.0 - leverage))[:, None]
    covariance = xtx_inverse @ design.T @ weighted_x @ xtx_inverse

    estimate_value = float(coefficients[1])
    standard_error = float(np.sqrt(covariance[1, 1]))
    degrees_freedom = len(data) - design.shape[1]
    statistic = estimate_value / standard_error
    p_value = float(2.0 * stats.t.sf(abs(statistic), degrees_freedom))
    return {
        "estimate": estimate_value,
        "std_error": standard_error,
        "p_value": p_value,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="CSV dataset")
    parser.add_argument("--output", required=True, help="Path for JSON output")
    args = parser.parse_args()

    result = estimate(pd.read_csv(args.data))
    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2)
        output_file.write("\n")


if __name__ == "__main__":
    main()
