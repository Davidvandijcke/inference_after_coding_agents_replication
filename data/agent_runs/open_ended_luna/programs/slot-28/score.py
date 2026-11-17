#!/usr/bin/env python3
"""Score the selected Pathways to Work subgroup analysis.

Estimates the treatment effect on six-month debt collections among participants
with some college or more, controlling for site, with HC2 robust standard error.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


def score(data_path: str) -> dict[str, float]:
    data = pd.read_csv(data_path)
    required = ["education", "treatment", "site", "debt_collections_m6"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    subgroup = data.loc[data["education"] >= 3, required].dropna().copy()
    if subgroup.empty:
        raise ValueError("No complete observations in the education >= 3 subgroup")

    site_indicators = pd.get_dummies(
        subgroup["site"], drop_first=True, dtype=float
    ).to_numpy()
    treatment = subgroup["treatment"].to_numpy(dtype=float)
    outcome = subgroup["debt_collections_m6"].to_numpy(dtype=float)
    design = np.column_stack(
        [np.ones(len(subgroup), dtype=float), treatment, site_indicators]
    )

    n, k = design.shape
    degrees_of_freedom = n - k
    if degrees_of_freedom <= 0 or np.linalg.matrix_rank(design) < k:
        raise ValueError("The subgroup data do not identify the site-adjusted model")

    bread = np.linalg.inv(design.T @ design)
    coefficients = bread @ design.T @ outcome
    residuals = outcome - design @ coefficients
    leverage = np.sum((design @ bread) * design, axis=1)
    if np.any(leverage >= 1):
        raise ValueError("HC2 variance is undefined for an observation with leverage 1")

    hc2_scores = design * (residuals / np.sqrt(1 - leverage))[:, None]
    covariance = bread @ (hc2_scores.T @ hc2_scores) @ bread

    estimate = float(coefficients[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    statistic = estimate / std_error
    p_value = float(2 * student_t.sf(abs(statistic), degrees_of_freedom))
    return {
        "estimate": estimate,
        "std_error": std_error,
        "p_value": p_value,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    result = score(args.data)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
