#!/usr/bin/env python3
"""Score the site-adjusted treatment effect on employment at 12 months."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="CSV dataset")
    parser.add_argument("--output", required=True, help="Path for JSON result")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = {"site", "treatment", "employed_m12"}
    missing = required.difference(data.columns)
    if missing:
        parser.error("dataset is missing required columns: " + ", ".join(sorted(missing)))
    if data[list(required)].isna().any().any():
        parser.error("required analysis columns must not contain missing values")

    # OLS with an intercept, treatment indicator, and site fixed effects.
    sites = sorted(data["site"].unique())
    design = np.column_stack(
        [
            np.ones(len(data)),
            data["treatment"].to_numpy(dtype=float),
            *(data["site"].eq(site).to_numpy(dtype=float) for site in sites[1:]),
        ]
    )
    outcome = data["employed_m12"].to_numpy(dtype=float)
    xtx_inv = np.linalg.inv(design.T @ design)
    coefficients = xtx_inv @ design.T @ outcome
    residuals = outcome - design @ coefficients

    # HC2 sandwich covariance: each squared residual is divided by (1 - h_ii).
    leverage = np.sum((design @ xtx_inv) * design, axis=1)
    meat = (design.T * (residuals**2 / (1.0 - leverage))) @ design
    covariance = xtx_inv @ meat @ xtx_inv
    estimate = float(coefficients[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    df_resid = len(data) - design.shape[1]
    p_value = float(2.0 * t.sf(abs(estimate / std_error), df_resid))

    result = {
        "estimate": estimate,
        "std_error": std_error,
        "p_value": p_value,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
