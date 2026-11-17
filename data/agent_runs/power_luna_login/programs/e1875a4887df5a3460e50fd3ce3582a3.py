#!/usr/bin/env python3
"""Estimate the site-adjusted 12-month employment treatment effect."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="CSV with the evaluation columns")
    parser.add_argument("--output", required=True, help="Path for the JSON result")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = {"site", "treatment", "employed_m12"}
    missing = required.difference(data.columns)
    if missing:
        parser.error("data is missing required columns: " + ", ".join(sorted(missing)))

    # OLS: 12-month employment on treatment and site indicators.
    site_indicators = pd.get_dummies(data["site"], drop_first=True, dtype=float).to_numpy()
    design = np.column_stack(
        [np.ones(len(data)), data["treatment"].to_numpy(dtype=float), site_indicators]
    )
    outcome = data["employed_m12"].to_numpy(dtype=float)

    xtx_inv = np.linalg.inv(design.T @ design)
    coefficients = xtx_inv @ design.T @ outcome
    residuals = outcome - design @ coefficients
    leverage = np.sum((design @ xtx_inv) * design, axis=1)
    weighted_design = design * (residuals**2 / (1.0 - leverage))[:, None]
    covariance = xtx_inv @ design.T @ weighted_design @ xtx_inv

    estimate = float(coefficients[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    degrees_freedom = len(data) - design.shape[1]
    p_value = float(2 * t.sf(abs(estimate / std_error), degrees_freedom))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {"estimate": estimate, "std_error": std_error, "p_value": p_value},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
