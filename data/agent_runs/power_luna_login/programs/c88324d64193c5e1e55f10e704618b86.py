#!/usr/bin/env python3
"""Score the selected Pathways to Work analysis on a same-schema dataset.

Chosen analysis: site-adjusted OLS treatment effect on 12-month employment
among participants who lived with at least one child at baseline. Inference is
the two-sided t-test with HC2 standard errors specified in CODEBOOK.md.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


def estimate(data: pd.DataFrame) -> dict[str, float]:
    required = {"site", "treatment", "parent", "employed_m12"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    sample = data.loc[data["parent"] == 1]
    if sample.empty:
        raise ValueError("No participants with parent == 1")

    site_dummies = pd.get_dummies(sample["site"], drop_first=True, dtype=float)
    design = np.column_stack(
        [np.ones(len(sample)), sample["treatment"].to_numpy(dtype=float), site_dummies.to_numpy()]
    )
    outcome = sample["employed_m12"].to_numpy(dtype=float)
    rank = np.linalg.matrix_rank(design)
    if rank != design.shape[1]:
        raise ValueError("Site-adjusted design matrix is rank deficient")

    xtx_inv = np.linalg.inv(design.T @ design)
    coefficients = xtx_inv @ design.T @ outcome
    residuals = outcome - design @ coefficients
    leverage = np.sum((design @ xtx_inv) * design, axis=1)

    # HC2 sandwich covariance: (X'X)^-1 X' diag(e_i^2 / (1-h_i)) X (X'X)^-1
    weighted_design = design * (residuals**2 / (1.0 - leverage))[:, None]
    covariance = xtx_inv @ design.T @ weighted_design @ xtx_inv
    treatment_estimate = float(coefficients[1])
    standard_error = float(np.sqrt(covariance[1, 1]))
    degrees_freedom = len(sample) - rank
    p_value = float(
        2 * student_t.sf(abs(treatment_estimate / standard_error), degrees_freedom)
    )
    return {
        "estimate": treatment_estimate,
        "std_error": standard_error,
        "p_value": p_value,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    result = estimate(pd.read_csv(args.data))
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
