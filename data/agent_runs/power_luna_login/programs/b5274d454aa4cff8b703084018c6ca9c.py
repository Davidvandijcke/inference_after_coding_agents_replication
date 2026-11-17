#!/usr/bin/env python3
"""Site-adjusted intention-to-treat effect on 12-month employment.

Usage: python score.py --data <path> --output <path>
Uses only the Python standard library. Output is a JSON object containing the
treatment coefficient, HC2 standard error, and two-sided t-test p-value.
"""

import argparse
import csv
import json
import math


def solve(matrix, vector):
    """Solve a small nonsingular linear system by Gauss-Jordan elimination."""
    n = len(vector)
    a = [list(map(float, matrix[i])) + [float(vector[i])] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(a[row][col]))
        if abs(a[pivot][col]) < 1e-12:
            raise ValueError("Design matrix is singular")
        a[col], a[pivot] = a[pivot], a[col]
        scale = a[col][col]
        a[col] = [v / scale for v in a[col]]
        for row in range(n):
            if row == col:
                continue
            scale = a[row][col]
            a[row] = [u - scale * v for u, v in zip(a[row], a[col])]
    return [a[i][-1] for i in range(n)]


def inverse(matrix):
    n = len(matrix)
    columns = [solve(matrix, [float(i == j) for i in range(n)]) for j in range(n)]
    return [[columns[j][i] for j in range(n)] for i in range(n)]


def beta_continued_fraction(a, b, x):
    """Continued fraction for the incomplete beta function (Numerical Recipes)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    tiny = 1e-300
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 10001):
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


def regularized_beta(x, a, b):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * beta_continued_fraction(a, b, x) / a
    return 1.0 - front * beta_continued_fraction(b, a, 1.0 - x) / b


def two_sided_t_pvalue(statistic, degrees_freedom):
    x = degrees_freedom / (degrees_freedom + statistic * statistic)
    return regularized_beta(x, degrees_freedom / 2.0, 0.5)


def estimate(rows):
    sites = sorted({row["site"] for row in rows})
    if len(sites) < 2:
        raise ValueError("At least two sites are required")
    # Intercept, treatment, then site indicators (first site is the reference).
    design = []
    outcome = []
    for row in rows:
        design.append([1.0, row["treatment"]] + [float(row["site"] == s) for s in sites[1:]])
        outcome.append(row["employed_m12"])
    n, k = len(rows), len(design[0])
    if n <= k:
        raise ValueError("Not enough observations for the site-adjusted model")

    xtx = [[sum(x[i] * x[j] for x in design) for j in range(k)] for i in range(k)]
    bread = inverse(xtx)
    xty = [sum(x[i] * y for x, y in zip(design, outcome)) for i in range(k)]
    coefficients = [sum(bread[i][j] * xty[j] for j in range(k)) for i in range(k)]

    meat = [[0.0] * k for _ in range(k)]
    for x, y in zip(design, outcome):
        residual = y - sum(x[j] * coefficients[j] for j in range(k))
        # HC2 uses each observation's OLS leverage adjustment.
        leverage = sum(x[i] * bread[i][j] * x[j] for i in range(k) for j in range(k))
        weight = residual * residual / (1.0 - leverage)
        for i in range(k):
            for j in range(k):
                meat[i][j] += weight * x[i] * x[j]
    covariance = [[
        sum(bread[i][a] * meat[a][b] * bread[b][j]
            for a in range(k) for b in range(k))
        for j in range(k)] for i in range(k)]
    coefficient = coefficients[1]
    standard_error = math.sqrt(covariance[1][1])
    p_value = two_sided_t_pvalue(abs(coefficient / standard_error), n - k)
    return {"estimate": coefficient, "std_error": standard_error, "p_value": p_value}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="CSV input dataset")
    parser.add_argument("--output", required=True, help="Path for JSON output")
    args = parser.parse_args()
    with open(args.data, newline="", encoding="utf-8") as source:
        rows = [{
            "site": int(row["site"]),
            "treatment": float(row["treatment"]),
            "employed_m12": float(row["employed_m12"]),
        } for row in csv.DictReader(source)]
    result = estimate(rows)
    with open(args.output, "w", encoding="utf-8") as destination:
        json.dump(result, destination, indent=2)
        destination.write("\n")


if __name__ == "__main__":
    main()
