#!/usr/bin/env python3
"""Score the site-adjusted intention-to-treat effect on 12-month employment."""

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import t


def score(data_path):
    data = pd.read_csv(data_path)
    outcome = data["employed_m12"].to_numpy(dtype=float)
    treatment = data["treatment"].to_numpy(dtype=float)

    # OLS with an intercept, treatment, and site indicators. The omitted site
    # is the reference category; randomization occurred within site.
    sites = sorted(data["site"].dropna().unique())
    columns = [np.ones(len(data)), treatment]
    columns.extend(data["site"].eq(site).to_numpy(dtype=float) for site in sites[1:])
    design = np.column_stack(columns)

    n, k = design.shape
    xtx_inv = np.linalg.inv(design.T @ design)
    coefficients = xtx_inv @ design.T @ outcome
    residuals = outcome - design @ coefficients
    leverage = np.sum((design @ xtx_inv) * design, axis=1)

    # HC2 covariance: scale each squared residual by 1 / (1 - leverage).
    adjusted = design * (residuals / np.sqrt(1.0 - leverage))[:, None]
    covariance = xtx_inv @ (adjusted.T @ adjusted) @ xtx_inv
    estimate = float(coefficients[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    p_value = float(2.0 * t.sf(abs(estimate / std_error), df=n - k))

    return {"estimate": estimate, "std_error": std_error, "p_value": p_value}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="CSV dataset with the evaluation columns")
    parser.add_argument("--output", required=True, help="Path for the JSON result")
    args = parser.parse_args()

    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(score(args.data), output_file)
        output_file.write("\n")


if __name__ == "__main__":
    main()
