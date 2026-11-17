#!/usr/bin/env python3
"""Score the Pathways to Work 12-month employment treatment effect.

Fits OLS of employed_m12 on treatment and site indicators, with an intercept,
and reports the treatment coefficient, its HC2 standard error, and a two-sided
t-test p-value using residual degrees of freedom.
"""

import argparse
import csv
import json
import sys

import numpy as np
from scipy.stats import t


def score(data_path):
    with open(data_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"site", "treatment", "employed_m12"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
        rows = list(reader)

    if not rows:
        raise ValueError("The input dataset has no rows")

    try:
        sites = np.asarray([int(row["site"]) for row in rows])
        treatment = np.asarray([float(row["treatment"]) for row in rows])
        outcome = np.asarray([float(row["employed_m12"]) for row in rows])
    except (TypeError, ValueError) as exc:
        raise ValueError("site, treatment, and employed_m12 must be numeric") from exc

    site_levels = np.unique(sites)
    # Intercept, treatment, and site indicators (first observed site omitted).
    columns = [np.ones(len(rows)), treatment]
    columns.extend((sites == site).astype(float) for site in site_levels[1:])
    x = np.column_stack(columns)
    n, k = x.shape
    if n <= k:
        raise ValueError("Not enough rows to estimate the model")

    xtx_inv = np.linalg.inv(x.T @ x)
    beta = xtx_inv @ x.T @ outcome
    residuals = outcome - x @ beta
    leverage = np.sum((x @ xtx_inv) * x, axis=1)
    if np.any(1.0 - leverage <= 0):
        raise ValueError("HC2 variance is undefined for this design")

    weighted_x = x * (residuals**2 / (1.0 - leverage))[:, None]
    covariance = xtx_inv @ (x.T @ weighted_x) @ xtx_inv
    estimate = float(beta[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    if std_error <= 0:
        raise ValueError("Treatment standard error is not positive")
    p_value = float(2.0 * t.sf(abs(estimate / std_error), df=n - k))
    return {"estimate": estimate, "std_error": std_error, "p_value": p_value}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Input CSV dataset")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    try:
        result = score(args.data)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
            f.write("\n")
    except (OSError, ValueError, np.linalg.LinAlgError) as exc:
        print(f"score.py: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
