#!/usr/bin/env python3
"""Run the fixed, selected Pathways to Work analysis.

Usage: python score.py --data data.csv --output result.json
Dependencies: numpy, scipy.

Estimand: assignment effect on food_insecure_m12 for all participants.
Model: OLS with intercept, treatment, and categorical site fixed effects.
Inference: HC2 standard error; two-sided Student t test, df = n - rank(X).
Effects are in probability units (multiply by 100 for percentage points).
The p-value is nominal, without adjustment for the original exploratory search.
This program performs no outcome, subgroup, or specification selection.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import t


def chosen_analysis(data_path):
    """Return exactly the three requested statistics for the fixed analysis."""
    columns = ("treatment", "site", "food_insecure_m12")
    with Path(data_path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = set(columns).difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
        observations = []
        for line, row in enumerate(reader, start=2):
            try:
                observations.append([float(row[name]) for name in columns])
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Missing or nonnumeric value on CSV line {line}") from exc
    if not observations:
        raise ValueError("The dataset contains no participants")
    values = np.asarray(observations, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("All required values must be finite and observed")
    treatment, site, outcome = values.T
    if not np.isin(treatment, [0, 1]).all():
        raise ValueError("treatment must be 0 or 1")
    if not np.isin(outcome, [0, 1]).all():
        raise ValueError("food_insecure_m12 must be 0 or 1")
    if not np.isin(site, [1, 2, 3, 4]).all():
        raise ValueError("site must be 1, 2, 3, or 4")

    # Reference site is the smallest site present; all participants are included.
    levels = np.unique(site)
    x = np.column_stack([np.ones(len(values)), treatment] +
                        [(site == level).astype(float) for level in levels[1:]])
    beta, _, rank, _ = np.linalg.lstsq(x, outcome, rcond=None)
    if rank != x.shape[1]:
        raise ValueError("Treatment effect is not identified after site adjustment")
    degrees_freedom = len(outcome) - rank
    if degrees_freedom <= 0:
        raise ValueError("Insufficient observations for inference")

    # HC2 uses squared residuals divided by (1 - leverage).
    bread = np.linalg.inv(x.T @ x)
    residuals = outcome - x @ beta
    leverage = np.einsum("ij,jk,ik->i", x, bread, x)
    if np.any(1 - leverage <= np.finfo(float).eps * 100):
        raise ValueError("HC2 is undefined for an observation with leverage one")
    residual_variance = residuals**2 / (1 - leverage)
    # Row 1 of (X'X)^(-1) X' maps outcomes to the treatment coefficient.
    treatment_weights = (bread @ x.T)[1]
    standard_error = float(np.sqrt(np.sum(treatment_weights**2 * residual_variance)))
    if not np.isfinite(standard_error) or standard_error <= 0:
        raise ValueError("A positive finite standard error is required for the t test")
    estimate = float(beta[1])
    p_value = float(2 * t.sf(abs(estimate / standard_error), df=degrees_freedom))
    return {"estimate": estimate, "std_error": standard_error, "p_value": p_value}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error("The output path must differ from the input dataset")
    try:
        result = chosen_analysis(args.data)
        with args.output.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, allow_nan=False)
            handle.write("\n")
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Analysis failed: {exc}\n")


if __name__ == "__main__":
    main()
