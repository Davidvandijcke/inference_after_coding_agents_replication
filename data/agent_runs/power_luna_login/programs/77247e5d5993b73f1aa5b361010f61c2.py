#!/usr/bin/env python3
"""Score the site-adjusted intention-to-treat effect on 12-month employment."""

import argparse
import csv
import json
import math


def invert(matrix):
    """Invert a small square matrix using Gauss-Jordan elimination."""
    size = len(matrix)
    work = [
        [float(value) for value in row]
        + [float(i == j) for j in range(size)]
        for i, row in enumerate(matrix)
    ]
    for col in range(size):
        pivot = max(range(col, size), key=lambda row: abs(work[row][col]))
        if abs(work[pivot][col]) < 1e-14:
            raise ValueError("The regression design matrix is singular")
        work[col], work[pivot] = work[pivot], work[col]
        scale = work[col][col]
        work[col] = [value / scale for value in work[col]]
        for row in range(size):
            if row == col:
                continue
            scale = work[row][col]
            work[row] = [
                value - scale * pivot_value
                for value, pivot_value in zip(work[row], work[col])
            ]
    return [row[size:] for row in work]


def beta_continued_fraction(a, b, x):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-300:
        d = 1e-300
    d = 1.0 / d
    result = d
    for iteration in range(1, 301):
        twice = 2 * iteration
        term = iteration * (b - iteration) * x / (
            (qam + twice) * (a + twice)
        )
        d = 1.0 + term * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + term / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        result *= d * c

        term = -(a + iteration) * (qab + iteration) * x / (
            (a + twice) * (qap + twice)
        )
        d = 1.0 + term * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + term / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        change = d * c
        result *= change
        if abs(change - 1.0) < 3e-14:
            break
    return result


def regularized_beta(a, b, x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    factor = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return factor * beta_continued_fraction(a, b, x) / a
    return 1.0 - factor * beta_continued_fraction(b, a, 1.0 - x) / b


def analyze(path):
    with open(path, newline="", encoding="utf-8") as source:
        records = list(csv.DictReader(source))
    required = {"site", "treatment", "employed_m12"}
    if not records or not required.issubset(records[0]):
        raise ValueError("Input must contain site, treatment, and employed_m12 columns")

    sites = sorted({row["site"] for row in records})
    reference_site = sites[0]
    design = []
    outcome = []
    for row in records:
        treatment = float(row["treatment"])
        if treatment not in (0.0, 1.0):
            raise ValueError("treatment must be coded 0/1")
        design.append(
            [1.0, treatment]
            + [float(row["site"] == site) for site in sites if site != reference_site]
        )
        outcome.append(float(row["employed_m12"]))

    n = len(outcome)
    width = len(design[0])
    xtx = [
        [sum(row[i] * row[j] for row in design) for j in range(width)]
        for i in range(width)
    ]
    xtx_inverse = invert(xtx)
    xty = [
        sum(row[i] * value for row, value in zip(design, outcome))
        for i in range(width)
    ]
    coefficients = [
        sum(xtx_inverse[i][j] * xty[j] for j in range(width))
        for i in range(width)
    ]

    # HC2 covariance: (X'X)^-1 X' diag(e_i^2 / (1 - h_i)) X (X'X)^-1.
    meat = [[0.0] * width for _ in range(width)]
    for row, value in zip(design, outcome):
        residual = value - sum(x * b for x, b in zip(row, coefficients))
        leverage = sum(
            row[i] * xtx_inverse[i][j] * row[j]
            for i in range(width)
            for j in range(width)
        )
        weight = residual * residual / (1.0 - leverage)
        for i in range(width):
            for j in range(width):
                meat[i][j] += weight * row[i] * row[j]
    covariance = [
        [
            sum(
                xtx_inverse[i][a] * meat[a][b] * xtx_inverse[b][j]
                for a in range(width)
                for b in range(width)
            )
            for j in range(width)
        ]
        for i in range(width)
    ]

    estimate = coefficients[1]
    std_error = math.sqrt(covariance[1][1])
    degrees_freedom = n - width
    statistic = estimate / std_error
    p_value = regularized_beta(
        degrees_freedom / 2.0,
        0.5,
        degrees_freedom / (degrees_freedom + statistic * statistic),
    )
    return {
        "estimate": estimate,
        "std_error": std_error,
        "p_value": p_value,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="CSV input dataset")
    parser.add_argument("--output", required=True, help="JSON output path")
    args = parser.parse_args()
    result = analyze(args.data)
    with open(args.output, "w", encoding="utf-8") as destination:
        json.dump(result, destination, indent=2)
        destination.write("\n")


if __name__ == "__main__":
    main()
