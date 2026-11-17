"""Estimate the full-sample treatment effect on employment at 12 months.

Usage: python score.py --data data.csv --output result.json
Requires numpy and scipy. Uses site-adjusted OLS, HC2 standard errors, and
a two-sided t-test with residual degrees of freedom. No outcome selection,
baseline adjustment, subgroup analysis, or multiple-testing adjustment is run.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import t


def score(data_path):
    with open(data_path, newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        required = {"site", "treatment", "employed_m12"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Data must contain site, treatment, and employed_m12.")
        rows = [(r["site"], float(r["treatment"]), float(r["employed_m12"]))
                for r in reader]
    if not rows:
        raise ValueError("Data contain no observations.")

    sites = np.array([row[0] for row in rows])
    treatment = np.array([row[1] for row in rows])
    outcome = np.array([row[2] for row in rows])
    if np.any(sites == ""):
        raise ValueError("Site must be observed for every participant.")
    if not np.all(np.isin(treatment, [0, 1])):
        raise ValueError("Treatment must be coded 0 or 1 without missing values.")
    if not np.all(np.isin(outcome, [0, 1])):
        raise ValueError("Employment must be coded 0 or 1 without missing values.")

    levels = np.unique(sites)
    design = np.column_stack(
        [np.ones(len(rows)), treatment]
        + [(sites == level).astype(float) for level in levels[1:]]
    )
    n, k = design.shape
    if n <= k or np.linalg.matrix_rank(design) != k:
        raise ValueError("The specified regression needs a full-rank design and positive residual degrees of freedom.")

    coefficients = np.linalg.lstsq(design, outcome, rcond=None)[0]
    bread = np.linalg.inv(design.T @ design)
    residual = outcome - design @ coefficients
    leverage = np.sum((design @ bread) * design, axis=1)
    if np.any(1 - leverage <= np.finfo(float).eps * 10):
        raise ValueError("HC2 standard errors are undefined at leverage one.")
    hc2_weights = residual**2 / (1 - leverage)
    covariance = bread @ (design.T @ (design * hc2_weights[:, None])) @ bread
    estimate = float(coefficients[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    if not np.isfinite(std_error) or std_error <= 0:
        raise ValueError("The treatment standard error must be finite and positive.")
    p_value = float(2 * t.sf(abs(estimate / std_error), df=n - k))
    return {"estimate": estimate, "std_error": std_error, "p_value": p_value}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error("Output must differ from the input dataset.")
    result = score(args.data)
    with args.output.open("w", encoding="utf-8") as destination:
        json.dump(result, destination, allow_nan=False)
        destination.write("\n")


if __name__ == "__main__":
    main()
