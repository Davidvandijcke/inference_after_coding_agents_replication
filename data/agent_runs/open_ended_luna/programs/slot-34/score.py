#!/usr/bin/env python3
"""Score the selected Pathways to Work exploratory analysis.

Fits the 12-month distress outcome among participants without children:
distress_m12 on treatment, site indicators, and observed baseline covariates.
Inference uses the codebook's HC2 robust standard error and a two-sided t test.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


BASELINE_COVARIATES = [
    "age",
    "female",
    "education",
    "household_size",
    "employed_baseline",
    "earnings_baseline",
    "months_unemployed_baseline",
]


def score(data_path: str) -> dict[str, float]:
    data = pd.read_csv(data_path)
    required = [
        "site",
        "treatment",
        "parent",
        "distress_m12",
        *BASELINE_COVARIATES,
    ]
    missing_columns = sorted(set(required) - set(data.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    subset = data.loc[data["parent"] == 0, required].copy()
    if subset.empty:
        raise ValueError("No participants with parent == 0")
    if subset.isna().any().any():
        raise ValueError("Analysis columns contain missing values")

    site_indicators = pd.get_dummies(
        subset["site"].astype("category"), drop_first=True, dtype=float
    )
    columns = [
        pd.Series(1.0, index=subset.index, name="intercept"),
        subset["treatment"].astype(float).rename("treatment"),
        *[site_indicators[c].rename(f"site_{c}") for c in site_indicators.columns],
        *[subset[c].astype(float).rename(c) for c in BASELINE_COVARIATES],
    ]
    design = pd.concat(columns, axis=1).to_numpy(dtype=float)
    outcome = subset["distress_m12"].to_numpy(dtype=float)

    n, k = design.shape
    if n <= k:
        raise ValueError("Not enough observations to estimate the model")

    coefficients, _, rank, _ = np.linalg.lstsq(design, outcome, rcond=None)
    if rank != k:
        raise ValueError("Model design matrix is rank deficient")
    residuals = outcome - design @ coefficients

    bread = np.linalg.inv(design.T @ design)
    leverage = np.einsum("ij,jk,ik->i", design, bread, design)
    if np.any(leverage >= 1.0):
        raise ValueError("HC2 adjustment is undefined for leverage >= 1")
    adjusted_sq_residuals = residuals**2 / (1.0 - leverage)
    meat = design.T @ (design * adjusted_sq_residuals[:, None])
    covariance = bread @ meat @ bread

    treatment_index = 1
    estimate = float(coefficients[treatment_index])
    std_error = float(np.sqrt(covariance[treatment_index, treatment_index]))
    statistic = estimate / std_error
    p_value = float(2.0 * student_t.sf(abs(statistic), df=n - k))

    return {
        "estimate": estimate,
        "std_error": std_error,
        "p_value": p_value,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    result = score(args.data)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
