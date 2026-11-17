#!/usr/bin/env python3
"""Score the parent-status difference in the 12-month employment effect."""

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="CSV with evaluation columns")
    parser.add_argument("--output", required=True, help="Path for JSON result")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = {"parent", "site", "treatment", "employed_m12"}
    missing = sorted(required.difference(data.columns))
    if missing:
        parser.error("missing required columns: " + ", ".join(missing))

    # Test whether the randomized treatment effect differs by parent status.
    # The interaction coefficient is the childless-minus-parent effect.
    parent = data["parent"].to_numpy(dtype=float)
    treatment = data["treatment"].to_numpy(dtype=float)
    if data["parent"].nunique() != 2 or data["treatment"].nunique() != 2:
        parser.error("data must include both parent groups and both treatment arms")
    site_dummies = pd.get_dummies(data["site"], drop_first=True, dtype=float)
    x = np.column_stack(
        [np.ones(len(data)), treatment, parent, treatment * (1.0 - parent),
         site_dummies.to_numpy(dtype=float)]
    )
    y = data["employed_m12"].to_numpy(dtype=float)
    n, k = x.shape
    if n <= k:
        parser.error("not enough observations to estimate the model")

    xtx_inv = np.linalg.inv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    residuals = y - x @ beta
    leverage = np.einsum("ij,jk,ik->i", x, xtx_inv, x)
    weighted_x = x * (residuals**2 / (1.0 - leverage))[:, None]
    covariance = xtx_inv @ x.T @ weighted_x @ xtx_inv
    estimate = float(beta[3])
    std_error = float(np.sqrt(covariance[3, 3]))
    p_value = float(2.0 * student_t.sf(abs(estimate / std_error), df=n - k))

    with open(args.output, "w", encoding="utf-8") as output:
        json.dump(
            {"estimate": estimate, "std_error": std_error, "p_value": p_value},
            output,
        )
        output.write("\n")


if __name__ == "__main__":
    main()
