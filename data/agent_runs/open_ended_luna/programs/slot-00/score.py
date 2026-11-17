#!/usr/bin/env python3
"""Score the chosen exploratory education-by-treatment contrast."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import t


def read_analysis_data(path):
    required = {"site", "treatment", "education", "distress_m12"}
    sites, treatment, low_education, outcome = [], [], [], []
    with Path(path).open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))
        for line, row in enumerate(reader, start=2):
            try:
                sites.append(row["site"])
                treatment.append(float(row["treatment"]))
                low_education.append(float(float(row["education"]) == 1.0))
                outcome.append(float(row["distress_m12"]))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid analysis value on CSV line {line}") from exc

    if not outcome:
        raise ValueError("The input data has no observations")
    if not all(value in (0.0, 1.0) for value in treatment):
        raise ValueError("treatment must contain only 0 and 1")
    if not all(np.isfinite(value) for value in treatment + low_education + outcome):
        raise ValueError("Analysis columns must contain finite numeric values")
    if not all(site != "" for site in sites):
        raise ValueError("site must be observed for every participant")

    return sites, np.asarray(treatment), np.asarray(low_education), np.asarray(outcome)


def analyze(path):
    sites, treatment, low_education, outcome = read_analysis_data(path)
    site_levels = sorted(set(sites))

    # Model: distress_m12 ~ treatment * low_education + site indicators.
    columns = [
        np.ones(len(outcome)),
        treatment,
        low_education,
        treatment * low_education,
    ]
    columns.extend(
        np.asarray([site == level for site in sites], dtype=float)
        for level in site_levels[1:]
    )
    design = np.column_stack(columns)
    rank = np.linalg.matrix_rank(design)
    if rank != design.shape[1]:
        raise ValueError("The chosen model is rank deficient for this dataset")

    n = len(outcome)
    degrees_freedom = n - rank
    if degrees_freedom <= 0:
        raise ValueError("The chosen model has no residual degrees of freedom")

    bread = np.linalg.inv(design.T @ design)
    coefficients = bread @ design.T @ outcome
    residuals = outcome - design @ coefficients
    leverage = np.einsum("ij,jk,ik->i", design, bread, design)
    if np.any(leverage >= 1.0):
        raise ValueError("HC2 variance is undefined for an observation with leverage 1")

    scaled_design = design * (residuals**2 / (1.0 - leverage))[:, None]
    meat = design.T @ scaled_design
    covariance = bread @ meat @ bread

    # The interaction is the difference in treatment effects for education=1
    # versus education>=2. Use the residual-df two-sided t test.
    index = 3
    estimate = float(coefficients[index])
    std_error = float(np.sqrt(covariance[index, index]))
    statistic = estimate / std_error
    p_value = float(2.0 * t.sf(abs(statistic), degrees_freedom))
    return {
        "estimate": estimate,
        "std_error": std_error,
        "p_value": p_value,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Estimate the site-adjusted 12-month distress interaction."
    )
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    result = analyze(args.data)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
