#!/usr/bin/env python3
"""Score the site-adjusted ITT effect on employment at 12 months.

Usage: python score.py --data <path> --output <path>
The output JSON contains the treatment coefficient, its HC2 standard error,
and the two-sided t-test p-value using residual degrees of freedom.
"""

import argparse
import csv
import json
import sys

import numpy as np
from scipy.stats import t as student_t


def estimate_effect(data_path):
    required = {"site", "treatment", "employed_m12"}
    with open(data_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))
        rows = list(reader)

    if len(rows) < 2:
        raise ValueError("At least two observations are required")

    try:
        sites = np.asarray([row["site"] for row in rows])
        treatment = np.asarray([float(row["treatment"]) for row in rows])
        outcome = np.asarray([float(row["employed_m12"]) for row in rows])
    except (TypeError, ValueError) as exc:
        raise ValueError("Required columns must contain numeric, non-missing values") from exc

    if not (np.isfinite(treatment).all() and np.isfinite(outcome).all()):
        raise ValueError("Treatment and outcome values must be finite")
    if not np.isin(treatment, [0.0, 1.0]).all():
        raise ValueError("treatment must contain only 0 and 1")
    site_levels = np.unique(sites)
    site_indicators = np.column_stack([sites == level for level in site_levels[1:]])
    design = np.column_stack((np.ones(len(rows)), treatment, site_indicators.astype(float)))

    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError("Treatment and site indicators do not identify the model")

    xtx_inverse = np.linalg.inv(design.T @ design)
    coefficients = xtx_inverse @ design.T @ outcome
    residuals = outcome - design @ coefficients
    leverage = np.einsum("ij,jk,ik->i", design, xtx_inverse, design)
    if np.any(leverage >= 1.0):
        raise ValueError("HC2 standard errors are undefined for a leverage value of 1")

    weighted_design = design * (residuals**2 / (1.0 - leverage))[:, None]
    covariance = xtx_inverse @ design.T @ weighted_design @ xtx_inverse
    estimate = float(coefficients[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    residual_df = len(rows) - design.shape[1]
    if residual_df <= 0 or std_error <= 0:
        raise ValueError("Insufficient residual degrees of freedom for inference")
    statistic = estimate / std_error
    p_value = float(2.0 * student_t.sf(abs(statistic), df=residual_df))
    return {"estimate": estimate, "std_error": std_error, "p_value": p_value}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV dataset")
    parser.add_argument("--output", required=True, help="Path for output JSON")
    args = parser.parse_args()

    try:
        result = estimate_effect(args.data)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")
    except (OSError, ValueError, np.linalg.LinAlgError) as exc:
        print(f"score.py: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
