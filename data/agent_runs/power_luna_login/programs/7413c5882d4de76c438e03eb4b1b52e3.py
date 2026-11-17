#!/usr/bin/env python3
"""Score the prespecified exploratory lead from a Pathways to Work dataset.

Fits debt_collections_m6 on treatment and site indicators among participants
employed at baseline, using HC2 robust standard errors.
"""

import argparse
import csv
import json
import sys

import numpy as np
from scipy.stats import t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    with open(args.data, newline="", encoding="utf-8") as f:
        rows = [row for row in csv.DictReader(f)
                if int(row["employed_baseline"]) == 1]

    if not rows:
        raise ValueError("No participants employed at baseline were found")

    outcome = np.asarray([float(row["debt_collections_m6"]) for row in rows])
    treatment = np.asarray([float(row["treatment"]) for row in rows])
    site = np.asarray([int(row["site"]) for row in rows])
    sites = sorted(set(site))
    columns = [np.ones(len(rows)), treatment]
    columns.extend((site == value).astype(float) for value in sites[1:])
    design = np.column_stack(columns)

    xtx_inverse = np.linalg.inv(design.T @ design)
    coefficient = xtx_inverse @ design.T @ outcome
    residual = outcome - design @ coefficient
    leverage = np.sum((design @ xtx_inverse) * design, axis=1)
    meat = design.T @ ((residual**2 / (1.0 - leverage))[:, None] * design)
    covariance = xtx_inverse @ meat @ xtx_inverse

    estimate = float(coefficient[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    degrees_freedom = len(rows) - design.shape[1]
    p_value = float(2.0 * t.sf(abs(estimate / std_error), degrees_freedom))

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({
            "estimate": estimate,
            "std_error": std_error,
            "p_value": p_value,
        }, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    try:
        main()
    except (KeyError, ValueError, np.linalg.LinAlgError) as error:
        print(f"score.py: {error}", file=sys.stderr)
        sys.exit(2)
