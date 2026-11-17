#!/usr/bin/env python3
"""Score the prespecified lead analysis: site-adjusted effect on 12-month employment."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


def score(data_path: Path) -> dict[str, float]:
    data = pd.read_csv(data_path)
    required = {"site", "treatment", "employed_m12"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    if data[list(required)].isna().any().any():
        raise ValueError("Required analysis columns must not contain missing values")

    # OLS with an intercept, site indicators, and the randomized treatment flag.
    site_indicators = pd.get_dummies(data["site"].astype(str), drop_first=True, dtype=float)
    treatment = data["treatment"].to_numpy(dtype=float)
    outcome = data["employed_m12"].to_numpy(dtype=float)
    x = np.column_stack((np.ones(len(data)), site_indicators.to_numpy(), treatment))
    if not np.isin(treatment, [0.0, 1.0]).all():
        raise ValueError("treatment must contain only 0 and 1")
    if not np.isin(outcome, [0.0, 1.0]).all():
        raise ValueError("employed_m12 must contain only 0 and 1")
    n, k = x.shape
    if n <= k or np.linalg.matrix_rank(x) < k:
        raise ValueError("Insufficient observations or rank-deficient design")

    xtx_inv = np.linalg.inv(x.T @ x)
    beta = xtx_inv @ x.T @ outcome
    residuals = outcome - x @ beta
    leverage = np.sum((x @ xtx_inv) * x, axis=1)
    # HC2 sandwich covariance: residual squared divided by (1 - leverage).
    meat = x.T @ ((residuals**2 / (1.0 - leverage))[:, None] * x)
    covariance = xtx_inv @ meat @ xtx_inv
    estimate = float(beta[-1])
    std_error = float(np.sqrt(covariance[-1, -1]))
    p_value = float(2.0 * student_t.sf(abs(estimate / std_error), df=n - k))
    return {"estimate": estimate, "std_error": std_error, "p_value": p_value}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Input CSV dataset")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON path")
    args = parser.parse_args()
    result = score(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
