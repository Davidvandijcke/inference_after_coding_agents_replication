#!/usr/bin/env python3
"""Estimate the assignment effect on 12-month employment.

Usage: python score.py --data data.csv --output result.json
Requires numpy and scipy. Fits OLS with an intercept, treatment, and site
indicators; uses HC2 standard errors and a two-sided residual-df t-test.
The returned p-value is for this fixed test, without a multiple-test adjustment.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import t


def analyze(data_path):
    with open(data_path, newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        required = {"site", "treatment", "employed_m12"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Data must contain site, treatment, and employed_m12")
        rows = [(row["site"], float(row["treatment"]),
                 float(row["employed_m12"])) for row in reader]

    if not rows or any(not row[0].strip() for row in rows):
        raise ValueError("Data must contain observations with nonmissing sites")
    sites = sorted({row[0] for row in rows})
    treatment = np.array([row[1] for row in rows], dtype=float)
    outcome = np.array([row[2] for row in rows], dtype=float)
    if not (np.isin(treatment, [0, 1]).all()
            and np.isin(outcome, [0, 1]).all()):
        raise ValueError("Treatment and employed_m12 must be observed binary values")

    design = np.column_stack(
        [np.ones(len(rows)), treatment]
        + [np.array([row[0] == site for row in rows], dtype=float)
           for site in sites[1:]]
    )
    n, k = design.shape
    if n <= k or np.linalg.matrix_rank(design) != k:
        raise ValueError("OLS requires a full-rank design and positive residual df")

    coefficients = np.linalg.lstsq(design, outcome, rcond=None)[0]
    bread = np.linalg.inv(design.T @ design)
    residuals = outcome - design @ coefficients
    leverage = np.sum((design @ bread) * design, axis=1)
    if np.any(1 - leverage <= np.finfo(float).eps * k):
        raise ValueError("HC2 is undefined for observations with unit leverage")
    treatment_weights = design @ bread[:, 1]
    variance = np.sum(treatment_weights**2 * residuals**2 / (1 - leverage))
    std_error = float(np.sqrt(variance))
    if not np.isfinite(std_error) or std_error <= 0:
        raise ValueError("A positive, finite HC2 standard error is required")

    estimate = float(coefficients[1])
    p_value = float(2 * t.sf(abs(estimate / std_error), df=n - k))
    return {"estimate": estimate, "std_error": std_error, "p_value": p_value}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error("The output path must differ from the input data path")
    result = analyze(args.data)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8")


if __name__ == "__main__":
    main()
