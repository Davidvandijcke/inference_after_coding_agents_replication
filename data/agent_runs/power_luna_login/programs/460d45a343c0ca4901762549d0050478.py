#!/usr/bin/env python3
"""Score the preselected Pathways to Work subgroup analysis.

Estimates the treatment effect on 6-month employment among participants who
were employed at baseline, adjusting for site indicators and using HC2 robust
standard errors. This script runs only this analysis; it does not search across
outcomes or subgroups.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t


def analyze(data_path: str) -> dict[str, float]:
    data = pd.read_csv(data_path)
    sample = data.loc[data["employed_baseline"] == 1].copy()
    if sample.empty or sample["treatment"].nunique() != 2:
        raise ValueError("The baseline-employed subgroup must contain both treatment arms.")

    sites = sorted(sample["site"].dropna().unique())
    columns = [np.ones(len(sample)), sample["treatment"].to_numpy(dtype=float)]
    columns.extend((sample["site"].to_numpy() == site).astype(float) for site in sites[1:])
    design = np.column_stack(columns)
    outcome = sample["employed_m6"].to_numpy(dtype=float)

    rank = np.linalg.matrix_rank(design)
    if rank != design.shape[1]:
        raise ValueError("The site-adjusted design matrix is rank deficient.")
    df = len(outcome) - rank
    if df <= 0:
        raise ValueError("Not enough observations to estimate the model.")

    bread = np.linalg.inv(design.T @ design)
    coefficients = bread @ design.T @ outcome
    residuals = outcome - design @ coefficients
    leverage = np.einsum("ij,jk,ik->i", design, bread, design)
    hc2_meat = design.T @ ((residuals**2 / (1.0 - leverage))[:, None] * design)
    covariance = bread @ hc2_meat @ bread

    estimate = float(coefficients[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    p_value = float(2 * t.sf(abs(estimate / std_error), df))
    return {"estimate": estimate, "std_error": std_error, "p_value": p_value}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="CSV dataset with the supplied columns")
    parser.add_argument("--output", required=True, help="Path for the JSON result")
    args = parser.parse_args()
    result = analyze(args.data)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
