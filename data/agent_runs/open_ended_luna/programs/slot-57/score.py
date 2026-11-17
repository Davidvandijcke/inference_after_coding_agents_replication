#!/usr/bin/env python3
"""Score the site-adjusted ITT effect on employment at month 6.

The fitted model is employed_m6 ~ treatment + site indicators. Inference uses
the HC2 sandwich standard error and a two-sided t test with residual degrees
of freedom, matching the evaluation codebook.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t


def read_analysis_columns(path):
    required = ("site", "treatment", "employed_m6")
    sites = []
    treatment = []
    outcome = []

    with open(path, "r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = [name for name in required if name not in (reader.fieldnames or [])]
        if missing:
            raise ValueError("Missing required columns: " + ", ".join(missing))

        for line_number, row in enumerate(reader, start=2):
            try:
                site = float(row["site"])
                assigned = float(row["treatment"])
                employed = float(row["employed_m6"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid analysis value on CSV line {line_number}") from exc
            if not np.isfinite([site, assigned, employed]).all():
                raise ValueError(f"Non-finite analysis value on CSV line {line_number}")
            sites.append(site)
            treatment.append(assigned)
            outcome.append(employed)

    if not outcome:
        raise ValueError("Input data has no participant rows")
    return np.asarray(sites), np.asarray(treatment), np.asarray(outcome)


def score(data_path):
    sites, treatment, outcome = read_analysis_columns(data_path)
    site_levels = np.unique(sites)

    # Intercept, treatment, and site indicators (first site is reference).
    columns = [np.ones(len(outcome)), treatment]
    columns.extend((sites == level).astype(float) for level in site_levels[1:])
    design = np.column_stack(columns)
    n, k = design.shape
    df_resid = n - k
    if df_resid <= 0:
        raise ValueError("Not enough participant rows to estimate the model")

    xtx_inverse = np.linalg.inv(design.T @ design)
    coefficients = xtx_inverse @ design.T @ outcome
    residuals = outcome - design @ coefficients
    leverage = np.einsum("ij,jk,ik->i", design, xtx_inverse, design)
    if np.any(1.0 - leverage <= 0.0):
        raise ValueError("HC2 variance is undefined for this input")

    hc2_weights = residuals**2 / (1.0 - leverage)
    meat = design.T @ (design * hc2_weights[:, None])
    covariance = xtx_inverse @ meat @ xtx_inverse
    estimate = float(coefficients[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    if not np.isfinite(std_error) or std_error <= 0.0:
        raise ValueError("Could not compute a positive finite HC2 standard error")
    statistic = estimate / std_error
    p_value = float(2.0 * student_t.sf(abs(statistic), df_resid))

    return {
        "estimate": estimate,
        "std_error": std_error,
        "p_value": p_value,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    result = score(args.data)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
