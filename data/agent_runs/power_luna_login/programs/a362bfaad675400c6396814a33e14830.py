#!/usr/bin/env python3
"""Score the site-adjusted intention-to-treat effect on 12-month employment.

Usage: python score.py --data <path> --output <path>

Requires NumPy and SciPy. The model is OLS of employed_m12 on treatment and
site indicators, with an HC2 sandwich standard error and a two-sided t test.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t


def score(data_path: Path) -> dict[str, float]:
    with data_path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"site", "treatment", "employed_m12"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
        rows = list(reader)

    if not rows:
        raise ValueError("The input dataset has no observations")

    try:
        site = np.asarray([float(row["site"]) for row in rows])
        treatment = np.asarray([float(row["treatment"]) for row in rows])
        outcome = np.asarray([float(row["employed_m12"]) for row in rows])
    except (TypeError, ValueError) as exc:
        raise ValueError("site, treatment and employed_m12 must be numeric") from exc

    if not (np.isfinite(site).all() and np.isfinite(treatment).all() and np.isfinite(outcome).all()):
        raise ValueError("Required columns must contain only finite values")
    if not np.isin(treatment, [0, 1]).all():
        raise ValueError("treatment must contain only 0 and 1")
    if not np.isin(outcome, [0, 1]).all():
        raise ValueError("employed_m12 must contain only 0 and 1")
    if len(np.unique(treatment)) != 2:
        raise ValueError("Both treatment groups must be represented")

    sites = np.unique(site)
    site_indicators = [(site == level).astype(float) for level in sites[1:]]
    design = np.column_stack([np.ones(len(rows)), treatment, *site_indicators])
    n, k = design.shape
    if n <= k:
        raise ValueError("Not enough observations for the site-adjusted model")

    bread = np.linalg.inv(design.T @ design)
    coefficients = bread @ design.T @ outcome
    residuals = outcome - design @ coefficients
    leverage = np.einsum("ij,jk,ik->i", design, bread, design)
    if np.any(1.0 - leverage <= 0):
        raise ValueError("HC2 variance is undefined for this design")
    weighted_design = design * (residuals**2 / (1.0 - leverage))[:, None]
    covariance = bread @ design.T @ weighted_design @ bread
    estimate = float(coefficients[1])
    std_error = float(math.sqrt(covariance[1, 1]))
    p_value = float(2.0 * student_t.sf(abs(estimate / std_error), df=n - k))
    return {"estimate": estimate, "std_error": std_error, "p_value": p_value}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Input CSV dataset")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON path")
    args = parser.parse_args()

    result = score(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
