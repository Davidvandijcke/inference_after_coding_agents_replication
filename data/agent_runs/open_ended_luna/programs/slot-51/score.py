#!/usr/bin/env python3
"""Score the 12-month debt-in-collections treatment effect.

Fits OLS of debt_collections_m12 on treatment and site indicators, then reports
the treatment estimate, HC2 standard error, and two-sided t-test p-value.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t


OUTCOME = "debt_collections_m12"


def estimate_effect(data: pd.DataFrame) -> dict[str, float]:
    required = {"site", "treatment", OUTCOME}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    sample = data[["site", "treatment", OUTCOME]].copy()
    if sample.isna().any().any():
        raise ValueError("Required analysis columns contain missing values")
    if not sample["treatment"].isin([0, 1]).all():
        raise ValueError("treatment must contain only 0 and 1")

    site_indicators = pd.get_dummies(
        sample["site"].astype("category"), drop_first=True, dtype=float
    ).to_numpy()
    treatment = sample["treatment"].to_numpy(dtype=float)
    outcome = sample[OUTCOME].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(sample)), treatment, site_indicators])

    if len(sample) <= design.shape[1]:
        raise ValueError("Not enough rows to estimate the model")

    xtx_inv = np.linalg.inv(design.T @ design)
    coefficients = xtx_inv @ (design.T @ outcome)
    residuals = outcome - design @ coefficients
    leverage = np.sum((design @ xtx_inv) * design, axis=1)
    hc2_meat = design.T @ (
        (residuals**2 / (1.0 - leverage))[:, None] * design
    )
    covariance = xtx_inv @ hc2_meat @ xtx_inv

    estimate = float(coefficients[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    df_resid = len(sample) - design.shape[1]
    p_value = float(2.0 * t.sf(abs(estimate / std_error), df_resid))

    return {
        "estimate": estimate,
        "std_error": std_error,
        "p_value": p_value,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV file")
    parser.add_argument("--output", required=True, help="Output JSON file")
    args = parser.parse_args()

    result = estimate_effect(pd.read_csv(args.data))
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
