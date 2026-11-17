#!/usr/bin/env python3
"""Score the site-adjusted 12-month employment effect in a Pathways dataset."""

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
        raise ValueError("Required analysis columns must not contain missing values")

    site_indicators = pd.get_dummies(
        data["site"].astype(str), drop_first=True, dtype=float
    ).to_numpy()
    treatment = data["treatment"].to_numpy(dtype=float)
    outcome = data["employed_m12"].to_numpy(dtype=float)
    design = np.column_stack((np.ones(len(data)), treatment, site_indicators))

    xtx_inverse = np.linalg.inv(design.T @ design)
    coefficients = xtx_inverse @ design.T @ outcome
    residuals = outcome - design @ coefficients
    leverage = np.sum((design @ xtx_inverse) * design, axis=1)
    weighted_design = (residuals**2 / (1.0 - leverage))[:, None] * design
    covariance = xtx_inverse @ (design.T @ weighted_design) @ xtx_inverse

    estimate = float(coefficients[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    degrees_freedom = len(data) - design.shape[1]
    if degrees_freedom <= 0 or std_error <= 0:
        raise ValueError("Insufficient data to estimate the treatment effect")
    p_value = float(
        2.0
        * student_t.sf(abs(estimate / std_error), df=degrees_freedom)
    )
    return estimate, std_error, p_value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV dataset")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    estimate, std_error, p_value = estimate_effect(pd.read_csv(args.data))
    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(
            {
                "estimate": estimate,
                "std_error": std_error,
                "p_value": p_value,
            },
            output_file,
            indent=2,
        )
        output_file.write("\n")


if __name__ == "__main__":
    main()
