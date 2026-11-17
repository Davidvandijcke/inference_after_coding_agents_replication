#!/usr/bin/env python3
"""Score the site-adjusted ITT effect on 12-month employment."""

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import t


def score(data_path):
    data = pd.read_csv(data_path)
    # Treatment coding with the first observed site as the reference site.
    site_dummies = pd.get_dummies(data["site"].astype("category"), drop_first=True, dtype=float)
    x = np.column_stack(
        [np.ones(len(data)), data["treatment"].to_numpy(dtype=float), site_dummies.to_numpy()]
    )
    y = data["employed_m12"].to_numpy(dtype=float)

    xtx_inv = np.linalg.inv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    residual = y - x @ beta
    leverage = np.einsum("ij,jk,ik->i", x, xtx_inv, x)
    # HC2 sandwich covariance.
    weighted_x = x * (residual**2 / (1.0 - leverage))[:, None]
    covariance = xtx_inv @ (x.T @ weighted_x) @ xtx_inv

    estimate = float(beta[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    degrees_freedom = len(y) - x.shape[1]
    statistic = estimate / std_error
    p_value = float(2 * t.sf(abs(statistic), degrees_freedom))
    return {"estimate": estimate, "std_error": std_error, "p_value": p_value}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="CSV dataset with the supplied columns")
    parser.add_argument("--output", required=True, help="Path for the JSON result")
    args = parser.parse_args()
    result = score(args.data)
    with open(args.output, "w", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
