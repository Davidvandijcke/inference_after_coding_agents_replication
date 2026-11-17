#!/usr/bin/env python3
"""Score the prespecified-for-this-script Pathways employment analysis.

Fits OLS of 12-month employment on treatment and site indicators, then reports
an HC2 robust standard error and a two-sided t-test for the treatment effect.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t


def load_columns(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        needed = {"site", "treatment", "employed_m12"}
        missing = needed.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
        rows = list(reader)
    if not rows:
        raise ValueError("Input dataset has no participant rows")
    try:
        sites = np.array([row["site"] for row in rows], dtype=str)
        treatment = np.array([float(row["treatment"]) for row in rows])
        outcome = np.array([float(row["employed_m12"]) for row in rows])
    except (TypeError, ValueError) as error:
        raise ValueError("site, treatment, and employed_m12 must be populated and numeric") from error
    if not np.isfinite(treatment).all() or not np.isfinite(outcome).all():
        raise ValueError("treatment and employed_m12 must be finite")
    return sites, treatment, outcome


def score(data_path):
    sites, treatment, outcome = load_columns(data_path)
    site_levels = sorted(set(sites))
    site_terms = np.column_stack([sites == level for level in site_levels[1:]]) if len(site_levels) > 1 else np.empty((len(sites), 0))
    design = np.column_stack((np.ones(len(sites)), treatment, site_terms.astype(float)))
    n, k = design.shape
    if n <= k:
        raise ValueError("Not enough rows to estimate the site-adjusted model")

    bread = np.linalg.pinv(design.T @ design)
    coefficients = bread @ design.T @ outcome
    residuals = outcome - design @ coefficients
    leverage = np.einsum("ij,jk,ik->i", design, bread, design)
    if np.any(1.0 - leverage <= 0):
        raise ValueError("HC2 variance is undefined for this dataset's design")
    weighted_design = design * (residuals**2 / (1.0 - leverage))[:, None]
    covariance = bread @ design.T @ weighted_design @ design @ bread
    estimate = float(coefficients[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    if std_error <= 0 or n - k <= 0:
        raise ValueError("Could not compute a valid treatment standard error")
    statistic = estimate / std_error
    p_value = float(2.0 * student_t.sf(abs(statistic), df=n - k))
    return {"estimate": estimate, "std_error": std_error, "p_value": p_value}


def main():
    parser = argparse.ArgumentParser(description="Estimate the site-adjusted 12-month employment effect.")
    parser.add_argument("--data", required=True, help="Input CSV dataset")
    parser.add_argument("--output", required=True, help="Path for the JSON result")
    args = parser.parse_args()
    result = score(args.data)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
