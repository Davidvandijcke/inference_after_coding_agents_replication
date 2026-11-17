#!/usr/bin/env python3
"""Score the prespecified 12-month employment treatment effect."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import t


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="CSV dataset path")
    parser.add_argument("--output", required=True, help="Path for the JSON result")
    args = parser.parse_args()

    data_path = Path(args.data)
    output_path = Path(args.output)
    if data_path.resolve() == output_path.resolve():
        parser.error("--output must not overwrite --data")

    with data_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"site", "treatment", "employed_m12"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            parser.error("dataset is missing columns: " + ", ".join(sorted(missing)))
        rows = list(reader)

    if not rows:
        parser.error("dataset has no observations")

    try:
        sites = [row["site"] for row in rows]
        treatment = np.array([float(row["treatment"]) for row in rows])
        outcome = np.array([float(row["employed_m12"]) for row in rows])
    except (TypeError, ValueError) as exc:
        parser.error(f"site, treatment, and employed_m12 must be observed numeric values: {exc}")

    if not np.isfinite(treatment).all() or not np.isfinite(outcome).all():
        parser.error("treatment and employed_m12 must contain only finite values")

    # OLS with an intercept, treatment, and site indicators (first site omitted).
    site_levels = sorted(set(sites))
    site_columns = [np.array([s == level for s in sites], dtype=float)
                    for level in site_levels[1:]]
    design = np.column_stack([np.ones(len(rows)), treatment, *site_columns])

    xtx_inverse = np.linalg.pinv(design.T @ design)
    beta = xtx_inverse @ design.T @ outcome
    residuals = outcome - design @ beta
    leverage = np.sum((design @ xtx_inverse) * design, axis=1)
    adjusted = residuals**2 / (1.0 - leverage)
    meat = design.T @ (adjusted[:, None] * design)
    covariance = xtx_inverse @ meat @ xtx_inverse

    rank = np.linalg.matrix_rank(design)
    df_resid = len(rows) - rank
    estimate = float(beta[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    p_value = float(2.0 * t.sf(abs(estimate / std_error), df_resid))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump({
            "estimate": estimate,
            "std_error": std_error,
            "p_value": p_value,
        }, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
