#!/usr/bin/env python3
"""Score the 12-month earnings effect among participants older than 40."""

import argparse
import csv
import json
import math
import sys


def solve(matrix, rhs):
    """Solve a small dense linear system with partial pivoting."""
    n = len(rhs)
    a = [list(map(float, row)) + [float(rhs[i])] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-12:
            raise ValueError("design matrix is singular")
        a[col], a[pivot] = a[pivot], a[col]
        scale = a[col][col]
        a[col] = [v / scale for v in a[col]]
        for row in range(n):
            if row == col:
                continue
            scale = a[row][col]
            a[row] = [v - scale * u for v, u in zip(a[row], a[col])]
    return [a[i][-1] for i in range(n)]


def inverse(matrix):
    n = len(matrix)
    columns = [solve(matrix, [1.0 if i == j else 0.0 for i in range(n)])
               for j in range(n)]
    return [[columns[j][i] for j in range(n)] for i in range(n)]


def beta_fraction(a, b, x):
    """Continued fraction used by the regularized incomplete beta function."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
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
    raise ArithmeticError("incomplete beta calculation did not converge")


def regularized_beta(x, a, b):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                     + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * beta_fraction(a, b, x) / a
    return 1.0 - front * beta_fraction(b, a, 1.0 - x) / b


def two_sided_t_pvalue(statistic, degrees_freedom):
    x = degrees_freedom / (degrees_freedom + statistic * statistic)
    return regularized_beta(x, degrees_freedom / 2.0, 0.5)


def score(data_path):
    with open(data_path, newline="", encoding="utf-8-sig") as handle:
        rows = [row for row in csv.DictReader(handle)
                if float(row["age"]) > 40]
    if not rows:
        raise ValueError("no participants older than 40 in input data")

    sites = sorted({int(row["site"]) for row in rows})
    # Intercept, treatment, and site indicators with the lowest site as reference.
    x = [[1.0, float(row["treatment"])] +
         [1.0 if int(row["site"]) == site else 0.0 for site in sites[1:]]
         for row in rows]
    y = [float(row["earnings_m12"]) for row in rows]
    n, p = len(x), len(x[0])
    if n <= p:
        raise ValueError("not enough observations for the site-adjusted model")

    xtx = [[sum(row[i] * row[j] for row in x) for j in range(p)]
           for i in range(p)]
    bread = inverse(xtx)
    xty = [sum(row[j] * value for row, value in zip(x, y)) for j in range(p)]
    coefficients = [sum(bread[i][j] * xty[j] for j in range(p))
                    for i in range(p)]
    residuals = [value - sum(a * b for a, b in zip(row, coefficients))
                 for row, value in zip(x, y)]
    leverage = [sum(row[i] * bread[i][j] * row[j]
                    for i in range(p) for j in range(p)) for row in x]

    meat = [[sum(row[i] * row[j] * residual**2 / (1.0 - h)
                 for row, residual, h in zip(x, residuals, leverage))
             for j in range(p)] for i in range(p)]
    covariance = [[sum(bread[i][a] * meat[a][b] * bread[b][j]
                       for a in range(p) for b in range(p))
                   for j in range(p)] for i in range(p)]
    standard_error = math.sqrt(covariance[1][1])
    degrees_freedom = n - p
    statistic = coefficients[1] / standard_error
    return {
        "estimate": coefficients[1],
        "std_error": standard_error,
        "p_value": two_sided_t_pvalue(statistic, degrees_freedom),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="input CSV dataset")
    parser.add_argument("--output", required=True, help="output JSON path")
    args = parser.parse_args()
    try:
        result = score(args.data)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")
    except (OSError, KeyError, ValueError, ArithmeticError) as error:
        print(f"score.py: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
