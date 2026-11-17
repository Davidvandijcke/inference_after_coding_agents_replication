#!/usr/bin/env python3
"""Score the site-adjusted 12-month employment effect (HC2 OLS)."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def _beta_continued_fraction(a, b, x):
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-300:
        d = 1e-300
    d = 1.0 / d
    h = d
    for m in range(1, 301):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-14:
            break
    return h


def _regularized_incomplete_beta(a, b, x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _beta_continued_fraction(a, b, x) / a
    return 1.0 - bt * _beta_continued_fraction(b, a, 1.0 - x) / b


def _two_sided_t_p_value(t_statistic, degrees_freedom):
    x = degrees_freedom / (degrees_freedom + t_statistic * t_statistic)
    return _regularized_incomplete_beta(degrees_freedom / 2.0, 0.5, x)


def score(data_path):
    required = {"site", "treatment", "employed_m12"}
    with Path(data_path).open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
        rows = list(reader)

    if not rows:
        raise ValueError("The input dataset has no observations")
    sites = [row["site"] for row in rows]
    treatment = np.asarray([float(row["treatment"]) for row in rows])
    outcome = np.asarray([float(row["employed_m12"]) for row in rows])
    levels = sorted(set(sites))
    site_columns = np.column_stack(
        [[1.0 if site == level else 0.0 for site in sites] for level in levels[1:]]
    ) if len(levels) > 1 else np.empty((len(rows), 0))
    design = np.column_stack((np.ones(len(rows)), treatment, site_columns))

    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("The treatment and site design matrix is rank deficient")
    bread = np.linalg.inv(design.T @ design)
    coefficients = bread @ design.T @ outcome
    residuals = outcome - design @ coefficients
    leverage = np.sum((design @ bread) * design, axis=1)
    hc2_meat = design.T @ ((residuals**2 / (1.0 - leverage))[:, None] * design)
    covariance = bread @ hc2_meat @ bread
    estimate = float(coefficients[1])
    std_error = math.sqrt(float(covariance[1, 1]))
    degrees_freedom = len(rows) - design.shape[1]
    if degrees_freedom <= 0 or not math.isfinite(std_error) or std_error <= 0.0:
        raise ValueError("Could not calculate a finite treatment standard error")
    p_value = _two_sided_t_p_value(abs(estimate / std_error), degrees_freedom)
    return {"estimate": estimate, "std_error": std_error, "p_value": p_value}


def main():
    parser = argparse.ArgumentParser(
        description="Estimate the site-adjusted 12-month employment effect using HC2 OLS."
    )
    parser.add_argument("--data", required=True, help="Input CSV dataset")
    parser.add_argument("--output", required=True, help="Path for output JSON")
    args = parser.parse_args()
    result = score(args.data)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as destination:
        json.dump(result, destination, indent=2)
        destination.write("\n")


if __name__ == "__main__":
    main()
