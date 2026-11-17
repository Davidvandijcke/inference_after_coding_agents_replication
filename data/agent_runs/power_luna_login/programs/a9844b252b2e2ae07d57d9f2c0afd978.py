#!/usr/bin/env python3
"""Score the 12-month employment effect using the Pathways trial convention."""

import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats


def estimate_effect(data: pd.DataFrame) -> tuple[float, float, float]:
    """OLS effect of assignment on 12-month employment, with site fixed effects
    and HC2 standard errors; return estimate, standard error, and two-sided
    t-test p-value.
    """
    required = {"site", "treatment", "employed_m12"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    if data[list(required)].isna().any().any():
        raise ValueError("Required analysis columns contain missing values")

    y = data["employed_m12"].to_numpy(dtype=float)
    treatment = data["treatment"].to_numpy(dtype=float)
    site = pd.get_dummies(data["site"], drop_first=True, dtype=float).to_numpy()
    design = np.column_stack((np.ones(len(data)), treatment, site))

    xtx_inverse = np.linalg.pinv(design.T @ design)
    coefficients = xtx_inverse @ design.T @ y
    residuals = y - design @ coefficients
    leverage = np.sum(design * (design @ xtx_inverse), axis=1)

    # HC2 sandwich covariance: adjust each squared residual by 1 - leverage.
    weighted_crossproduct = design.T @ (design * (residuals**2 / (1.0 - leverage))[:, None])
    covariance = xtx_inverse @ weighted_crossproduct @ xtx_inverse
    estimate = float(coefficients[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    degrees_freedom = len(y) - np.linalg.matrix_rank(design)
    if degrees_freedom <= 0 or std_error <= 0:
        raise ValueError("Unable to compute inference for this dataset")
    p_value = float(2 * stats.t.sf(abs(estimate / std_error), degrees_freedom))
    return estimate, std_error, p_value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Input CSV dataset")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    estimate, std_error, p_value = estimate_effect(pd.read_csv(args.data))
    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(
            {"estimate": estimate, "std_error": std_error, "p_value": p_value},
            output_file,
        )
        output_file.write("\n")


if __name__ == "__main__":
    main()
