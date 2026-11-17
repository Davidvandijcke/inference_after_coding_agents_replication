#!/usr/bin/env python3
"""Score the 12-month employment ITT effect using site-adjusted OLS and HC2."""

import argparse
import csv
import json
import math


def inverse(matrix):
    """Invert a nonsingular square matrix by Gauss-Jordan elimination."""
    n = len(matrix)
    work = [list(map(float, row)) + [float(i == j) for j in range(n)]
            for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(work[row][col]))
        if abs(work[pivot][col]) < 1e-14:
            raise ValueError("Regression design matrix is singular")
        work[col], work[pivot] = work[pivot], work[col]
        scale = work[col][col]
        work[col] = [value / scale for value in work[col]]
        for row in range(n):
            if row != col:
                scale = work[row][col]
                work[row] = [a - scale * b
                             for a, b in zip(work[row], work[col])]
    return [row[n:] for row in work]


def beta_fraction(a, b, x):
    """Continued fraction used to evaluate the regularized incomplete beta."""
    tiny, eps, limit = 1e-300, 3e-14, 300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    result = d
    for m in range(1, limit + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        result *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) < eps:
            return result
    raise ArithmeticError("Incomplete beta calculation did not converge")


def regularized_beta(x, a, b):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    factor = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                      + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return factor * beta_fraction(a, b, x) / a
    return 1.0 - factor * beta_fraction(b, a, 1.0 - x) / b


def score(data_path):
    with open(data_path, newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError("Input data has no observations")
    required = {"site", "treatment", "employed_m12"}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))

    sites = [row["site"] for row in rows]
    site_levels = sorted(set(sites))
    # Intercept, treatment assignment, and site indicators (first site omitted).
    design = []
    outcome = []
    for row, site in zip(rows, sites):
        design.append([1.0, float(row["treatment"])] +
                      [float(site == level) for level in site_levels[1:]])
        outcome.append(float(row["employed_m12"]))

    n, k = len(design), len(design[0])
    if n <= k:
        raise ValueError("Insufficient observations for the regression")
    xtx = [[sum(x[i] * x[j] for x in design) for j in range(k)]
           for i in range(k)]
    xtx_inv = inverse(xtx)
    xty = [sum(x[i] * y for x, y in zip(design, outcome)) for i in range(k)]
    coefficients = [sum(xtx_inv[i][j] * xty[j] for j in range(k))
                    for i in range(k)]
    residuals = [y - sum(a * b for a, b in zip(x, coefficients))
                 for x, y in zip(design, outcome)]

    # HC2: X' diag(e_i^2 / (1 - h_ii)) X, with h_ii = x_i'(X'X)^-1 x_i.
    meat = [[0.0] * k for _ in range(k)]
    for x, residual in zip(design, residuals):
        leverage = sum(x[i] * xtx_inv[i][j] * x[j]
                       for i in range(k) for j in range(k))
        if leverage >= 1.0:
            raise ValueError("HC2 is undefined for an observation with unit leverage")
        weight = residual * residual / (1.0 - leverage)
        for i in range(k):
            for j in range(k):
                meat[i][j] += weight * x[i] * x[j]
    covariance = [[sum(xtx_inv[i][a] * meat[a][b] * xtx_inv[b][j]
                       for a in range(k) for b in range(k))
                   for j in range(k)] for i in range(k)]

    estimate = coefficients[1]
    std_error = math.sqrt(covariance[1][1])
    df = n - k
    t_stat = estimate / std_error
    # Two-sided Student t tail: I_{df/(df+t^2)}(df/2, 1/2).
    p_value = regularized_beta(df / (df + t_stat * t_stat), df / 2.0, 0.5)
    return {"estimate": estimate, "std_error": std_error, "p_value": p_value}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="CSV input dataset")
    parser.add_argument("--output", required=True, help="Path for JSON output")
    args = parser.parse_args()
    result = score(args.data)
    with open(args.output, "w", encoding="utf-8") as destination:
        json.dump(result, destination, indent=2)
        destination.write("\n")


if __name__ == "__main__":
    main()
