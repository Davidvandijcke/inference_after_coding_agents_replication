#!/usr/bin/env python3
"""Score the site-adjusted 12-month employment treatment effect."""

import argparse
import csv
import json
import math
from pathlib import Path


def _inverse(matrix):
    """Invert a small square matrix by Gauss-Jordan elimination."""
    n = len(matrix)
    work = [list(map(float, row)) + [float(i == j) for j in range(n)]
            for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(work[row][col]))
        if abs(work[pivot][col]) < 1e-12:
            raise ValueError("design matrix is singular")
        work[col], work[pivot] = work[pivot], work[col]
        scale = work[col][col]
        work[col] = [value / scale for value in work[col]]
        for row in range(n):
            if row != col:
                scale = work[row][col]
                work[row] = [a - scale * b
                             for a, b in zip(work[row], work[col])]
    return [row[n:] for row in work]


def _betacf(a, b, x):
    """Continued fraction used to evaluate the regularized beta function."""
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


def _regularized_beta(x, a, b):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                     + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _two_sided_t_pvalue(statistic, degrees_freedom):
    x = degrees_freedom / (degrees_freedom + statistic * statistic)
    return _regularized_beta(x, degrees_freedom / 2.0, 0.5)


def analyze(data_path):
    with open(data_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"site", "treatment", "employed_m12"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("data must include site, treatment, employed_m12")
        rows = list(reader)

    if not rows:
        raise ValueError("data contains no observations")
    sites = sorted({row["site"] for row in rows})
    # Intercept, treatment, and site indicators (first site is the reference).
    design, outcome = [], []
    for row in rows:
        site = row["site"]
        design.append([1.0, float(row["treatment"])] +
                      [float(site == level) for level in sites[1:]])
        outcome.append(float(row["employed_m12"]))

    n, k = len(design), len(design[0])
    if n <= k:
        raise ValueError("not enough observations to estimate the model")
    xtx = [[sum(x[i] * x[j] for x in design) for j in range(k)]
           for i in range(k)]
    xtx_inv = _inverse(xtx)
    xty = [sum(x[i] * y for x, y in zip(design, outcome)) for i in range(k)]
    beta = [sum(xtx_inv[i][j] * xty[j] for j in range(k))
            for i in range(k)]

    # HC2 sandwich covariance: (X'X)^-1 X' diag(e_i^2/(1-h_i)) X (X'X)^-1.
    meat = [[0.0] * k for _ in range(k)]
    for x, y in zip(design, outcome):
        residual = y - sum(a * b for a, b in zip(x, beta))
        leverage = sum(x[i] * xtx_inv[i][j] * x[j]
                       for i in range(k) for j in range(k))
        weight = residual * residual / (1.0 - leverage)
        for i in range(k):
            for j in range(k):
                meat[i][j] += weight * x[i] * x[j]
    covariance = [[sum(xtx_inv[i][a] * meat[a][b] * xtx_inv[b][j]
                       for a in range(k) for b in range(k))
                    for j in range(k)] for i in range(k)]
    estimate = beta[1]
    std_error = math.sqrt(covariance[1][1])
    p_value = _two_sided_t_pvalue(estimate / std_error, n - k)
    return {"estimate": estimate, "std_error": std_error, "p_value": p_value}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="input CSV path")
    parser.add_argument("--output", required=True, help="output JSON path")
    args = parser.parse_args()
    result = analyze(args.data)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, separators=(",", ":"))
        handle.write("\n")


if __name__ == "__main__":
    main()
