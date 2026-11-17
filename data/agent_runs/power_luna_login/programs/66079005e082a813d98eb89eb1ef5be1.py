#!/usr/bin/env python3
"""Score the site-adjusted intention-to-treat effect on 12-month employment.

Uses OLS with an intercept, treatment indicator, and site indicators, with
HC2 heteroskedasticity-robust standard errors and a two-sided t test.
Only the fixed analysis below is run; there is no outcome or subgroup search.
"""

import argparse
import csv
import json
import math
from pathlib import Path


def _beta_continued_fraction(a, b, x):
    """Continued fraction used by the regularized incomplete beta function."""
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
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(a, b, x) / a
    return 1.0 - front * _beta_continued_fraction(b, a, 1.0 - x) / b


def _two_sided_t_pvalue(t_stat, degrees_freedom):
    x = degrees_freedom / (degrees_freedom + t_stat * t_stat)
    return _regularized_beta(x, degrees_freedom / 2.0, 0.5)


def _invert(matrix):
    n = len(matrix)
    augmented = [list(row) + [float(i == j) for j in range(n)]
                 for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(augmented[r][col]))
        if abs(augmented[pivot][col]) < 1e-14:
            raise ValueError("Design matrix is singular")
        augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        scale = augmented[col][col]
        augmented[col] = [v / scale for v in augmented[col]]
        for row in range(n):
            if row == col:
                continue
            scale = augmented[row][col]
            augmented[row] = [u - scale * v
                              for u, v in zip(augmented[row], augmented[col])]
    return [row[n:] for row in augmented]


def score(data_path):
    with open(data_path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"site", "treatment", "employed_m12"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))
        records = list(reader)

    sites = sorted({row["site"] for row in records})
    if len(sites) < 2:
        raise ValueError("At least two sites are required for site indicators")
    rows, outcomes = [], []
    for record in records:
        site = record["site"]
        treatment = float(record["treatment"])
        outcome = float(record["employed_m12"])
        rows.append([1.0, treatment] + [float(site == level) for level in sites[1:]])
        outcomes.append(outcome)

    n, k = len(rows), len(rows[0])
    if n <= k:
        raise ValueError("Not enough observations to estimate the model")
    xtx = [[sum(row[i] * row[j] for row in rows) for j in range(k)]
           for i in range(k)]
    bread = _invert(xtx)
    xty = [sum(row[i] * y for row, y in zip(rows, outcomes)) for i in range(k)]
    beta = [sum(bread[i][j] * xty[j] for j in range(k)) for i in range(k)]
    residuals = [y - sum(v * b for v, b in zip(row, beta))
                 for row, y in zip(rows, outcomes)]
    leverage = [sum(row[i] * bread[i][j] * row[j]
                    for i in range(k) for j in range(k)) for row in rows]

    meat = [[sum(row[i] * row[j] * residual ** 2 / (1.0 - h)
                 for row, residual, h in zip(rows, residuals, leverage))
             for j in range(k)] for i in range(k)]
    covariance = [[sum(bread[i][a] * meat[a][b] * bread[b][j]
                       for a in range(k) for b in range(k))
                   for j in range(k)] for i in range(k)]
    estimate = beta[1]
    std_error = math.sqrt(covariance[1][1])
    p_value = _two_sided_t_pvalue(abs(estimate / std_error), n - k)
    return {"estimate": estimate, "std_error": std_error, "p_value": p_value}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()
    result = score(args.data)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
