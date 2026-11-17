#!/usr/bin/env python3
"""Score the baseline-unemployed, 12-month employment treatment effect."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t


def score(data_path):
    data = pd.read_csv(data_path)
    sample = data.loc[data["employed_baseline"] == 0]

    outcome = sample["employed_m12"].to_numpy(dtype=float)
    treatment = sample["treatment"].to_numpy(dtype=float)
    site_indicators = pd.get_dummies(
        sample["site"].astype("category"), drop_first=True, dtype=float
    ).to_numpy()
    design = np.column_stack((np.ones(len(sample)), treatment, site_indicators))

    bread = np.linalg.pinv(design.T @ design)
    coefficients = bread @ design.T @ outcome
    residuals = outcome - design @ coefficients
    leverage = np.einsum("ij,jk,ik->i", design, bread, design)

    # HC2 sandwich covariance: each squared residual is divided by (1 - h_ii).
    adjusted_residual_sq = residuals**2 / (1.0 - leverage)
    meat = design.T @ (adjusted_residual_sq[:, None] * design)
    covariance = bread @ meat @ bread

    estimate = float(coefficients[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    degrees_freedom = len(sample) - np.linalg.matrix_rank(design)
    p_value = float(2 * t.sf(abs(estimate / std_error), degrees_freedom))
    return {
        "estimate": estimate,
        "std_error": std_error,
        "p_value": p_value,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV dataset")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    result = score(args.data)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
