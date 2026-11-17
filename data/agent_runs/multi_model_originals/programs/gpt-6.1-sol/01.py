#!/usr/bin/env python3
"""Run the frozen, exploratory Pathways to Work analysis.

Usage: python score.py --data data.csv --output result.json

Chosen estimand: intention-to-treat effect on training_m12 among participants
at site 4. Fit OLS with an intercept and treatment. Site indicators have no
variation within this subgroup and add no regressors. Use HC2 standard errors
and a two-sided Student t-test with the OLS residual degrees of freedom.

The estimate is a proportion difference (0.10 means 10 percentage points).
The p-value is nominal, without a correction for the exploratory selection.
This script never selects outcomes, sites, follow-up times, or covariates.

Dependencies: Python 3, numpy, scipy.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import t


def chosen_analysis(data_path):
    treatment = []
    outcome = []
    with open(data_path, newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        required = {"site", "treatment", "training_m12"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))
        for line, row in enumerate(reader, start=2):
            site = float(row["site"])
            if not np.isfinite(site):
                raise ValueError(f"Non-finite site on CSV line {line}")
            if site != 4:
                continue
            assignment = float(row["treatment"])
            enrolled = float(row["training_m12"])
            if assignment not in (0, 1) or enrolled not in (0, 1):
                raise ValueError(
                    f"Treatment and training_m12 must be 0 or 1 on CSV line {line}"
                )
            treatment.append(assignment)
            outcome.append(enrolled)

    treatment = np.asarray(treatment, dtype=float)
    y = np.asarray(outcome, dtype=float)
    if any(np.count_nonzero(treatment == arm) < 2 for arm in (0, 1)):
        raise ValueError("Site 4 must contain at least two participants per treatment arm")

    x = np.column_stack((np.ones(len(y)), treatment))
    bread = np.linalg.inv(x.T @ x)
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    residual = y - x @ beta
    leverage = np.einsum("ij,jk,ik->i", x, bread, x)
    hc2_weights = residual**2 / (1.0 - leverage)
    covariance = bread @ (x.T @ (hc2_weights[:, None] * x)) @ bread
    estimate = float(beta[1])
    std_error = float(np.sqrt(max(0.0, covariance[1, 1])))
    if std_error == 0:
        raise ValueError("HC2 standard error is zero; the requested t-test is undefined")
    p_value = float(2.0 * t.sf(abs(estimate / std_error), df=len(y) - 2))
    return {"estimate": estimate, "std_error": std_error, "p_value": p_value}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Input participant CSV")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON file")
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error("--output must be different from --data")
    result = chosen_analysis(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
