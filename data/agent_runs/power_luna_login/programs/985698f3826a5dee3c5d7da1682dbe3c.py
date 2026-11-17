#!/usr/bin/env python3
"""Score the site-adjusted 12-month employment treatment effect."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import t


def score(data_path: Path) -> dict[str, float]:
    with data_path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("data file has no participant rows")

    required = {"site", "treatment", "employed_m12"}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"missing required columns: {', '.join(sorted(missing))}")

    sites = [row["site"] for row in rows]
    treatments = np.asarray([float(row["treatment"]) for row in rows])
    outcome = np.asarray([float(row["employed_m12"]) for row in rows])
    if not np.isfinite(treatments).all() or not np.isfinite(outcome).all():
        raise ValueError("treatment and employed_m12 must be observed numeric values")

    site_levels = list(dict.fromkeys(sites))
    site_indicators = np.column_stack(
        [[1.0 if site == level else 0.0 for site in sites]
         for level in site_levels[1:]]
    ) if len(site_levels) > 1 else np.empty((len(rows), 0))
    design = np.column_stack((np.ones(len(rows)), treatments, site_indicators))

    xtx_inverse = np.linalg.inv(design.T @ design)
    coefficients = xtx_inverse @ design.T @ outcome
    residuals = outcome - design @ coefficients
    leverage = np.einsum("ij,jk,ik->i", design, xtx_inverse, design)
    hc2_meat = design.T @ ((residuals**2 / (1.0 - leverage))[:, None] * design)
    covariance = xtx_inverse @ hc2_meat @ xtx_inverse

    estimate = float(coefficients[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    degrees_freedom = len(rows) - np.linalg.matrix_rank(design)
    statistic = estimate / std_error
    p_value = float(2.0 * t.sf(abs(statistic), degrees_freedom))
    return {
        "estimate": estimate,
        "std_error": std_error,
        "p_value": p_value,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate the site-adjusted effect on 12-month employment."
    )
    parser.add_argument("--data", required=True, type=Path, help="input CSV path")
    parser.add_argument("--output", required=True, type=Path, help="output JSON path")
    args = parser.parse_args()
    result = score(args.data)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
