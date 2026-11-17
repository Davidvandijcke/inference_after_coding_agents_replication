#!/usr/bin/env python3
"""Score the Pathways to Work effect on employment at 12 months.

Fits OLS of employed_m12 on treatment and site indicators, and reports the
treatment coefficient, HC2 standard error, and two-sided t-test p-value.
"""

import argparse
import csv
import json
import sys

import numpy as np
from scipy.stats import t as student_t


def estimate(data_path):
    with open(data_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"site", "treatment", "employed_m12"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
        rows = list(reader)

    if not rows:
        raise ValueError("The input dataset has no observations")

    try:
        sites = np.array([row["site"] for row in rows])
        treatment = np.array([float(row["treatment"]) for row in rows])
        outcome = np.array([float(row["employed_m12"]) for row in rows])
    except (TypeError, ValueError) as exc:
        raise ValueError("site, treatment, and employed_m12 must contain valid values") from exc

    site_levels = sorted(set(sites.tolist()))
    site_indicators = np.column_stack(
        [(sites == level).astype(float) for level in site_levels[1:]]
    ) if len(site_levels) > 1 else np.empty((len(rows), 0))
    design = np.column_stack((np.ones(len(rows)), treatment, site_indicators))

    n, k = design.shape
    if n <= k or np.linalg.matrix_rank(design) < k:
        raise ValueError("The design matrix is rank deficient or has no residual degrees of freedom")

    bread = np.linalg.inv(design.T @ design)
    coefficients = bread @ design.T @ outcome
    residuals = outcome - design @ coefficients
    leverage = np.einsum("ij,jk,ik->i", design, bread, design)
    hc2_meat = (design * (residuals**2 / (1.0 - leverage))[:, None]).T @ design
    covariance = bread @ hc2_meat @ bread

    estimate = float(coefficients[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    if not np.isfinite(std_error) or std_error <= 0:
        raise ValueError("Could not compute a positive finite treatment standard error")
    statistic = estimate / std_error
    p_value = float(2.0 * student_t.sf(abs(statistic), df=n - k))
    return {"estimate": estimate, "std_error": std_error, "p_value": p_value}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV dataset")
    parser.add_argument("--output", required=True, help="Output JSON file")
    args = parser.parse_args()

    try:
        result = estimate(args.data)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")
    except (OSError, ValueError, np.linalg.LinAlgError) as exc:
        print(f"score.py: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
