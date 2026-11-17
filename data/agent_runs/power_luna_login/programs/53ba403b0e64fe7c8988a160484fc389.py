#!/usr/bin/env python3
"""Score the site-adjusted ITT effect on employment at 12 months.

Usage: python score.py --data <path> --output <path>
"""

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


def estimate_effect(data: pd.DataFrame) -> tuple[float, float, float]:
    """OLS of employed_m12 on treatment and site indicators, with HC2 SE."""
    required = {"site", "treatment", "employed_m12"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    if data[list(required)].isna().any().any():
        raise ValueError("Required analysis columns contain missing values")

    site = data["site"].to_numpy()
    treatment = data["treatment"].to_numpy(dtype=float)
    outcome = data["employed_m12"].to_numpy(dtype=float)
    site_levels = np.sort(np.unique(site))
    if len(site_levels) < 2:
        raise ValueError("At least two sites are required for site adjustment")

    # Intercept, treatment, and indicators for all but the first site.
    design = np.column_stack(
        [np.ones(len(data)), treatment]
        + [(site == level).astype(float) for level in site_levels[1:]]
    )
    n, k = design.shape
    if n <= k or np.linalg.matrix_rank(design) != k:
        raise ValueError("The analysis design matrix is not full rank")

    xtx_inverse = np.linalg.inv(design.T @ design)
    coefficients = xtx_inverse @ design.T @ outcome
    residuals = outcome - design @ coefficients

    # HC2 sandwich covariance: residual squared divided by (1 - leverage).
    leverage = np.sum((design @ xtx_inverse) * design, axis=1)
    if np.any(1.0 - leverage <= 0):
        raise ValueError("HC2 variance is undefined for this design")
    weighted_design = design * (residuals**2 / (1.0 - leverage))[:, None]
    covariance = xtx_inverse @ design.T @ weighted_design @ xtx_inverse

    estimate = float(coefficients[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    if std_error <= 0:
        raise ValueError("Treatment standard error is not positive")
    p_value = float(2.0 * student_t.sf(abs(estimate / std_error), df=n - k))
    return estimate, std_error, p_value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    estimate, std_error, p_value = estimate_effect(data)
    result = {
        "estimate": estimate,
        "std_error": std_error,
        "p_value": p_value,
    }
    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2)
        output_file.write("\n")


if __name__ == "__main__":
    main()
