#!/usr/bin/env python3
"""Score the site-adjusted intention-to-treat effect on 12-month employment."""

import argparse
import csv
import json
import math


def inverse(matrix):
    """Invert a small square matrix using Gauss-Jordan elimination."""
    n = len(matrix)
    augmented = [
        list(map(float, row)) + [float(i == j) for j in range(n)]
        for i, row in enumerate(matrix)
    ]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(augmented[row][col]))
        if abs(augmented[pivot][col]) < 1e-12:
            raise ValueError("design matrix is singular")
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


def dot(left, right):
    return sum(a * b for a, b in zip(left, right))


def beta_continued_fraction(a, b, x):
    """Continued fraction used by the regularized incomplete beta function."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    tiny = 1e-300
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    value = d
    for m in range(1, 201):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        value *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        value *= delta
        if abs(delta - 1.0) < 3e-14:
            break
    return value


def regularized_beta(x, a, b):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    factor = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return factor * beta_continued_fraction(a, b, x) / a
    return 1.0 - factor * beta_continued_fraction(b, a, 1.0 - x) / b


def analyze(path):
    with open(path, newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        required = {"site", "treatment", "employed_m12"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError("missing required columns: " + ", ".join(sorted(missing)))
        rows = list(reader)

    if not rows:
        raise ValueError("data contains no observations")
    sites = sorted({int(row["site"]) for row in rows})
    # OLS with an intercept, treatment indicator, and site fixed effects.
    design = []
    outcomes = []
    for row in rows:
        site = int(row["site"])
        design.append(
            [1.0, float(row["treatment"])]
            + [float(site == level) for level in sites[1:]]
        )
        outcomes.append(float(row["employed_m12"]))

    n, k = len(outcomes), len(design[0])
    if n <= k:
        raise ValueError("not enough observations to estimate the model")
    xtx = [
        [sum(row[i] * row[j] for row in design) for j in range(k)]
        for i in range(k)
    ]
    bread = inverse(xtx)
    xty = [
        sum(row[i] * outcome for row, outcome in zip(design, outcomes))
        for i in range(k)
    ]
    coefficients = [dot(row, xty) for row in bread]
    residuals = [
        outcome - dot(row, coefficients)
        for row, outcome in zip(design, outcomes)
    ]

    # HC2 sandwich covariance: (X'X)^-1 X' diag(e_i^2/(1-h_i)) X (X'X)^-1.
    meat = [[0.0] * k for _ in range(k)]
    for row, residual in zip(design, residuals):
        leverage = dot(row, [dot(bread_row, row) for bread_row in bread])
        weight = residual * residual / (1.0 - leverage)
        for i in range(k):
            for j in range(k):
                meat[i][j] += weight * row[i] * row[j]
    bread_meat = [
        [sum(bread[i][u] * meat[u][v] for u in range(k)) for v in range(k)]
        for i in range(k)
    ]
    covariance = [
        [sum(bread_meat[i][v] * bread[j][v] for v in range(k)) for j in range(k)]
        for i in range(k)
    ]

    estimate = coefficients[1]
    std_error = math.sqrt(covariance[1][1])
    if std_error <= 0.0:
        raise ValueError("treatment coefficient has no estimable standard error")
    df = n - k
    t_stat = estimate / std_error
    p_value = regularized_beta(df / (df + t_stat * t_stat), df / 2.0, 0.5)
    return {"estimate": estimate, "std_error": std_error, "p_value": p_value}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="input CSV path")
    parser.add_argument("--output", required=True, help="output JSON path")
    args = parser.parse_args()
    result = analyze(args.data)
    with open(args.output, "w", encoding="utf-8") as destination:
        json.dump(result, destination)
        destination.write("\n")


if __name__ == "__main__":
    main()
