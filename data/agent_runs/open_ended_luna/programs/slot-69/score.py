#!/usr/bin/env python3
"""Score the 12-month distress-by-baseline-unemployment-duration analysis."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


def score(data_path: Path) -> dict[str, float]:
    data = pd.read_csv(data_path)
    required = {
        "site",
        "treatment",
        "months_unemployed_baseline",
        "distress_m12",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    # The duration field is zero for participants who were employed at baseline.
    # Estimate the linear duration gradient among those unemployed at baseline.
    sample = data.loc[data["months_unemployed_baseline"] > 0].copy()
    if sample.empty:
        raise ValueError("No participants were unemployed at baseline")

    treatment = sample["treatment"].to_numpy(dtype=float)
    duration_years = (
        sample["months_unemployed_baseline"].to_numpy(dtype=float) / 12.0
    )
    duration_years -= duration_years.mean()
    site_indicators = pd.get_dummies(
        sample["site"], drop_first=True, dtype=float
    ).to_numpy()

    # The treatment-by-duration coefficient is the change in the treatment
    # effect for each additional year unemployed at baseline.
    design = np.column_stack(
        [
            np.ones(len(sample)),
            treatment,
            duration_years,
            treatment * duration_years,
            site_indicators,
        ]
    )
    outcome = sample["distress_m12"].to_numpy(dtype=float)
    n, k = design.shape
    if n <= k or np.linalg.matrix_rank(design) < k:
        raise ValueError("Insufficient observations or rank deficient design matrix")

    xtx_inverse = np.linalg.inv(design.T @ design)
    coefficients = xtx_inverse @ design.T @ outcome
    residuals = outcome - design @ coefficients
    leverage = np.einsum("ij,jk,ik->i", design, xtx_inverse, design)
    hc2_denominator = 1.0 - leverage
    if np.any(hc2_denominator <= 0):
        raise ValueError("HC2 covariance is undefined for this design")

    weighted_design = design * (residuals**2 / hc2_denominator)[:, None]
    meat = design.T @ weighted_design
    covariance = xtx_inverse @ meat @ xtx_inverse

    interaction_index = 3
    estimate = float(coefficients[interaction_index])
    std_error = float(np.sqrt(covariance[interaction_index, interaction_index]))
    degrees_freedom = n - k
    statistic = estimate / std_error
    p_value = float(2.0 * student_t.sf(abs(statistic), degrees_freedom))

    return {
        "estimate": estimate,
        "std_error": std_error,
        "p_value": p_value,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate the selected 12-month distress duration interaction."
    )
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    result = score(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
