#!/usr/bin/env python3
"""Score the 12-month employment effect using site-adjusted OLS and HC2 SEs."""

import argparse
import csv
import json
import math


def invert(matrix):
    n = len(matrix)
    aug = [list(map(float, row)) + [float(i == j) for j in range(n)]
           for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-14:
            raise ValueError("Design matrix is singular")
        aug[col], aug[pivot] = aug[pivot], aug[col]
        scale = aug[col][col]
        aug[col] = [v / scale for v in aug[col]]
        for row in range(n):
            if row != col:
                scale = aug[row][col]
                aug[row] = [a - scale * b for a, b in zip(aug[row], aug[col])]
    return [row[n:] for row in aug]


def betacf(a, b, x):
    """Continued fraction for the incomplete beta function."""
    max_iter, eps, tiny = 300, 3e-14, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
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
        if abs(delta - 1.0) < eps:
            break
    return h


def regularized_beta(x, a, b):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                     + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * betacf(a, b, x) / a
    return 1.0 - front * betacf(b, a, 1.0 - x) / b


def analyze(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("Input data has no rows")
    sites = sorted({int(row["site"]) for row in rows})
    if len(sites) < 2:
        raise ValueError("At least two sites are required")

    # Intercept, treatment, and site indicators, matching the codebook.
    design = []
    outcome = []
    for row in rows:
        site = int(row["site"])
        design.append([1.0, float(row["treatment"])]
                      + [float(site == s) for s in sites[1:]])
        outcome.append(float(row["employed_m12"]))
    n, k = len(design), len(design[0])

    xtx = [[sum(x[i] * x[j] for x in design) for j in range(k)]
           for i in range(k)]
    inv = invert(xtx)
    xty = [sum(x[i] * y for x, y in zip(design, outcome)) for i in range(k)]
    beta = [sum(inv[i][j] * xty[j] for j in range(k)) for i in range(k)]
    residuals = [y - sum(x[j] * beta[j] for j in range(k))
                 for x, y in zip(design, outcome)]

    meat = [[0.0] * k for _ in range(k)]
    for x, residual in zip(design, residuals):
        leverage = sum(x[i] * inv[i][j] * x[j]
                       for i in range(k) for j in range(k))
        weight = residual * residual / (1.0 - leverage)
        for i in range(k):
            for j in range(k):
                meat[i][j] += weight * x[i] * x[j]
    temp = [[sum(inv[i][a] * meat[a][j] for a in range(k))
             for j in range(k)] for i in range(k)]
    covariance = [[sum(temp[i][a] * inv[a][j] for a in range(k))
                   for j in range(k)] for i in range(k)]
    estimate = beta[1]
    std_error = math.sqrt(covariance[1][1])
    df = n - k
    t_stat = abs(estimate / std_error)
    p_value = regularized_beta(df / (df + t_stat * t_stat), df / 2.0, 0.5)
    return {"estimate": estimate, "std_error": std_error, "p_value": p_value}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="CSV with the codebook columns")
    parser.add_argument("--output", required=True, help="Path for JSON result")
    args = parser.parse_args()
    result = analyze(args.data)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
