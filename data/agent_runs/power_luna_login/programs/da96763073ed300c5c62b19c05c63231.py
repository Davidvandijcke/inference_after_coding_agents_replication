#!/usr/bin/env python3
"""Site-adjusted intention-to-treat effect on employment at 12 months."""

import argparse
import csv
import json
import math


def invert(matrix):
    n = len(matrix)
    augmented = [
        [float(value) for value in row]
        + [float(i == j) for j in range(n)]
        for i, row in enumerate(matrix)
    ]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(augmented[row][col]))
        if abs(augmented[pivot][col]) < 1e-12:
            raise ValueError("Design matrix is singular")
        augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        scale = augmented[col][col]
        augmented[col] = [value / scale for value in augmented[col]]
        for row in range(n):
            if row == col:
                continue
            scale = augmented[row][col]
            augmented[row] = [
                value - scale * pivot_value
                for value, pivot_value in zip(augmented[row], augmented[col])
            ]
    return [row[n:] for row in augmented]


def beta_fraction(a, b, x):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = 1e-300 if abs(d) < 1e-300 else d
    d = 1.0 / d
    result = d
    for iteration in range(1, 301):
        twice = 2 * iteration
        term = iteration * (b - iteration) * x / ((qam + twice) * (a + twice))
        d = 1.0 + term * d
        d = 1e-300 if abs(d) < 1e-300 else d
        c = 1.0 + term / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1.0 / d
        result *= d * c
        term = -(a + iteration) * (qab + iteration) * x / (
            (a + twice) * (qap + twice)
        )
        d = 1.0 + term * d
        d = 1e-300 if abs(d) < 1e-300 else d
        c = 1.0 + term / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) < 3e-14:
            break
    return result


def regularized_beta(a, b, x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    factor = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return factor * beta_fraction(a, b, x) / a
    return 1.0 - factor * beta_fraction(b, a, 1.0 - x) / b


def analyze(path):
    with open(path, newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError("Input data has no rows")
    required = {"site", "treatment", "employed_m12"}
    if not required.issubset(rows[0]):
        raise ValueError("Input is missing required columns: site, treatment, employed_m12")

    sites = sorted({row["site"] for row in rows})
    # Treatment plus an intercept and site indicators; omit one site as reference.
    design = [
        [1.0, float(row["treatment"])]
        + [float(row["site"] == site) for site in sites[1:]]
        for row in rows
    ]
    outcome = [float(row["employed_m12"]) for row in rows]
    n, p = len(rows), len(design[0])

    xtx = [
        [sum(x[i] * x[j] for x in design) for j in range(p)]
        for i in range(p)
    ]
    bread = invert(xtx)
    xty = [sum(x[i] * y for x, y in zip(design, outcome)) for i in range(p)]
    coefficients = [
        sum(bread[i][j] * xty[j] for j in range(p)) for i in range(p)
    ]

    meat = [[0.0] * p for _ in range(p)]
    for x, y in zip(design, outcome):
        residual = y - sum(value * coefficient for value, coefficient in zip(x, coefficients))
        leverage = sum(
            x[i] * bread[i][j] * x[j]
            for i in range(p)
            for j in range(p)
        )
        weight = residual * residual / (1.0 - leverage)
        for i in range(p):
            for j in range(p):
                meat[i][j] += weight * x[i] * x[j]

    covariance = [
        [
            sum(
                bread[i][a] * meat[a][b] * bread[b][j]
                for a in range(p)
                for b in range(p)
            )
            for j in range(p)
        ]
        for i in range(p)
    ]
    estimate = coefficients[1]
    std_error = math.sqrt(covariance[1][1])
    statistic = estimate / std_error
    degrees_freedom = n - p
    p_value = regularized_beta(
        degrees_freedom / 2.0,
        0.5,
        degrees_freedom / (degrees_freedom + statistic * statistic),
    )
    return {"estimate": estimate, "std_error": std_error, "p_value": p_value}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="CSV dataset path")
    parser.add_argument("--output", required=True, help="JSON output path")
    args = parser.parse_args()
    result = analyze(args.data)
    with open(args.output, "w", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
