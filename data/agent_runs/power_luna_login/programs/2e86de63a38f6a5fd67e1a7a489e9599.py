#!/usr/bin/env python3
"""Score the prespecified lead analysis: site-adjusted effect on 12-month employment."""

import argparse
import csv
import json
import math
from pathlib import Path


def matrix_inverse(matrix):
    """Invert a small square matrix with Gauss-Jordan elimination."""
    n = len(matrix)
    augmented = [list(map(float, row)) + [float(i == j) for j in range(n)]
                 for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(augmented[row][col]))
        if abs(augmented[pivot][col]) < 1e-12:
            raise ValueError("Regression design matrix is singular")
        augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        scale = augmented[col][col]
        augmented[col] = [v / scale for v in augmented[col]]
        for row in range(n):
            if row != col:
                scale = augmented[row][col]
                augmented[row] = [a - scale * b for a, b in
                                  zip(augmented[row], augmented[col])]
    return [row[n:] for row in augmented]


def beta_continued_fraction(a, b, x):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    tiny = 1e-300
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 10000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-14:
            return h
    raise ArithmeticError("Incomplete beta calculation did not converge")


def regularized_incomplete_beta(x, a, b):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                     + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * beta_continued_fraction(a, b, x) / a
    return 1.0 - front * beta_continued_fraction(b, a, 1.0 - x) / b


def score(data_path):
    with open(data_path, newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        required = {"site", "treatment", "employed_m12"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("Data must include site, treatment, and employed_m12 columns")
        rows = list(reader)

    if not rows:
        raise ValueError("Data contains no observations")
    sites = sorted({row["site"] for row in rows})
    n = len(rows)
    design = []
    outcome = []
    for row in rows:
        treatment = float(row["treatment"])
        y = float(row["employed_m12"])
        site_dummies = [float(row["site"] == site) for site in sites[1:]]
        design.append([1.0, treatment] + site_dummies)
        outcome.append(y)

    k = len(design[0])
    df = n - k
    if df <= 0:
        raise ValueError("Not enough observations for the regression")

    xtx = [[sum(row[i] * row[j] for row in design) for j in range(k)]
           for i in range(k)]
    xtx_inverse = matrix_inverse(xtx)
    xty = [sum(row[i] * y for row, y in zip(design, outcome)) for i in range(k)]
    coefficients = [sum(xtx_inverse[i][j] * xty[j] for j in range(k))
                    for i in range(k)]

    meat = [[0.0 for _ in range(k)] for _ in range(k)]
    for row, y in zip(design, outcome):
        residual = y - sum(a * b for a, b in zip(row, coefficients))
        fitted_leverage = sum(
            row[i] * xtx_inverse[i][j] * row[j]
            for i in range(k) for j in range(k)
        )
        if fitted_leverage >= 1.0:
            raise ValueError("HC2 variance is undefined for a leverage of 1")
        weight = residual * residual / (1.0 - fitted_leverage)
        for i in range(k):
            for j in range(k):
                meat[i][j] += weight * row[i] * row[j]

    covariance = [[
        sum(xtx_inverse[i][a] * meat[a][b] * xtx_inverse[b][j]
            for a in range(k) for b in range(k))
        for j in range(k)] for i in range(k)]
    estimate = coefficients[1]
    std_error = math.sqrt(covariance[1][1])
    t_statistic = estimate / std_error
    beta_x = df / (df + t_statistic * t_statistic)
    p_value = regularized_incomplete_beta(beta_x, df / 2.0, 0.5)
    return {"estimate": estimate, "std_error": std_error, "p_value": p_value}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV dataset")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()
    result = score(args.data)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
