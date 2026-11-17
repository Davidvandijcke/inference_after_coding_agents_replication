#!/usr/bin/env python3
"""Estimate the intention-to-treat effect on employment at 12 months.

Usage: python score.py --data data.csv --output result.json
Dependencies: numpy, scipy.

The sole analysis is OLS of employed_m12 on an intercept, treatment, and
indicators for observed sites (one omitted). There are no additional baseline
covariates or subgroup restrictions. Inference uses HC2 standard errors and a
two-sided Student t test with n - rank(X) degrees of freedom. The JSON p_value
is the unadjusted p-value for this fixed test, not a multiple-testing p-value.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import t


def score(data_path):
    """Run only the specified full-sample regression and return three scalars."""
    treatment, outcome, sites = [], [], []
    with open(data_path, newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        required = {"treatment", "site", "employed_m12"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Input requires treatment, site, and employed_m12 columns")
        for row in reader:
            treatment.append(float(row["treatment"]))
            outcome.append(float(row["employed_m12"]))
            sites.append(row["site"].strip())

    treatment = np.asarray(treatment, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    if not len(outcome) or any(site == "" for site in sites):
        raise ValueError("Input must have observations and nonmissing sites")
    if not np.isin(treatment, [0.0, 1.0]).all():
        raise ValueError("treatment must be observed and binary (0/1)")
    if not np.isin(outcome, [0.0, 1.0]).all():
        raise ValueError("employed_m12 must be observed and binary (0/1)")

    sites = np.asarray(sites)
    levels = sorted(set(sites))
    columns = [np.ones(len(outcome)), treatment]
    columns.extend((sites == level).astype(float) for level in levels[1:])
    design = np.column_stack(columns)
    n, k = design.shape
    if n <= k or np.linalg.matrix_rank(design) != k:
        raise ValueError("Regression requires full column rank and positive residual df")

    # X+ gives coefficients and the HC2 sandwich without inverting X'X.
    inverse = np.linalg.pinv(design)
    coefficients = inverse @ outcome
    residuals = outcome - design @ coefficients
    leverage = np.sum(design * inverse.T, axis=1)
    if np.any(1.0 - leverage <= np.finfo(float).eps):
        raise ValueError("HC2 is undefined for an observation with unit leverage")
    hc2_variances = residuals**2 / (1.0 - leverage)
    variance = np.sum(inverse[1] ** 2 * hc2_variances)
    std_error = float(np.sqrt(variance))
    if not np.isfinite(std_error) or std_error <= 0:
        raise ValueError("Treatment standard error must be finite and positive")
    estimate = float(coefficients[1])
    p_value = float(2.0 * t.sf(abs(estimate / std_error), df=n - k))
    return {"estimate": estimate, "std_error": std_error, "p_value": p_value}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error("--output must differ from --data")
    result = score(args.data)
    with args.output.open("w", encoding="utf-8") as destination:
        json.dump(result, destination, indent=2, allow_nan=False)
        destination.write("\n")


if __name__ == "__main__":
    main()
