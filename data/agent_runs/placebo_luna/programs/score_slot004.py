#!/usr/bin/env python3
"""Score the randomized-trial effect on follow-up action using block-adjusted OLS."""

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import t


OUTCOME = "took_action_fu"


def score(data_path):
    data = pd.read_csv(data_path)
    required = {"treatment", "block", OUTCOME}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    observed = data[["treatment", "block", OUTCOME]].dropna().copy()
    if observed["treatment"].nunique() != 2:
        raise ValueError("Both treatment groups must have observed outcome data")

    # Intercept, treatment indicator, and block indicators (first block omitted).
    block_dummies = pd.get_dummies(observed["block"], drop_first=True, dtype=float)
    x = np.column_stack(
        [np.ones(len(observed)), observed["treatment"].to_numpy(dtype=float),
         block_dummies.to_numpy(dtype=float)]
    )
    y = observed[OUTCOME].to_numpy(dtype=float)
    n, k = x.shape
    if n <= k:
        raise ValueError("Insufficient observed rows to estimate the model")

    bread = np.linalg.inv(x.T @ x)
    beta = bread @ x.T @ y
    residuals = y - x @ beta
    leverage = np.einsum("ij,jk,ik->i", x, bread, x)
    hc2_meat = x.T @ (x * (residuals**2 / (1.0 - leverage))[:, None])
    covariance = bread @ hc2_meat @ bread
    estimate = float(beta[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    statistic = estimate / std_error
    p_value = float(2.0 * t.sf(abs(statistic), df=n - k))
    return {
        "estimate": estimate,
        "std_error": std_error,
        "p_value": p_value,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="CSV file with trial data")
    parser.add_argument("--output", required=True, help="Path for JSON result")
    args = parser.parse_args()
    result = score(args.data)
    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2)
        output_file.write("\n")


if __name__ == "__main__":
    main()
