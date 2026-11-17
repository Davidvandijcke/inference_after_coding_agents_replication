#!/usr/bin/env python3
"""Score the selected age 36–40 follow-up analysis.

Fits OLS of predicted_employment_3y_fu on treatment and block indicators
among participants in age_group 3, using HC2 standard errors.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t


OUTCOME = "predicted_employment_3y_fu"


def score(data_path):
    data = pd.read_csv(data_path)
    required = {"age_group", "treatment", "block", OUTCOME}
    missing_columns = required.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing_columns))}")

    sample = data.loc[data["age_group"] == 3, ["treatment", "block", OUTCOME]].dropna()
    if sample.empty:
        raise ValueError("No complete observations in age_group 3")

    # Intercept, treatment, and block fixed effects (one category omitted).
    block_effects = pd.get_dummies(sample["block"], drop_first=True, dtype=float)
    x = np.column_stack(
        [np.ones(len(sample)), sample["treatment"].to_numpy(dtype=float), block_effects.to_numpy()]
    )
    y = sample[OUTCOME].to_numpy(dtype=float)
    rank = np.linalg.matrix_rank(x)
    if rank < 2 or len(y) <= rank:
        raise ValueError("Insufficient observations or treatment variation for the regression")

    bread = np.linalg.pinv(x.T @ x)
    beta = bread @ x.T @ y
    residuals = y - x @ beta
    leverage = np.einsum("ij,jk,ik->i", x, bread, x)
    if np.any(1 - leverage <= 0):
        raise ValueError("HC2 variance is undefined for this sample")
    weighted_x = x * (residuals**2 / (1 - leverage))[:, None]
    covariance = bread @ (x.T @ weighted_x) @ bread
    estimate = float(beta[1])
    std_error = float(np.sqrt(max(covariance[1, 1], 0.0)))
    if std_error == 0:
        p_value = 0.0 if estimate != 0 else 1.0
    else:
        statistic = estimate / std_error
        p_value = float(2 * t.sf(abs(statistic), df=len(y) - rank))

    return {"estimate": estimate, "std_error": std_error, "p_value": p_value}


def main():
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
