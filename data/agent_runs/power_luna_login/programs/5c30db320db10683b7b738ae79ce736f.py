#!/usr/bin/env python3
"""Score the site-adjusted ITT effect on employment at 12 months.

Usage: python score.py --data <path> --output <path>

Fits OLS of employed_m12 on treatment and site indicators, then reports the
treatment coefficient, its HC2 standard error, and a two-sided t-test p-value.
This program runs only this analysis; it does not search outcomes or subgroups.
It uses only the Python standard library.
"""

import argparse
import csv
import json
import math
from pathlib import Path


def matmul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(len(b)))
             for j in range(len(b[0]))] for i in range(len(a))]


def transpose(a):
    return [list(col) for col in zip(*a)]


def inverse(a):
    n = len(a)
    aug = [list(map(float, row)) + [float(i == j) for j in range(n)]
           for i, row in enumerate(a)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            raise ValueError("Design matrix is singular")
        aug[col], aug[pivot] = aug[pivot], aug[col]
        scale = aug[col][col]
        aug[col] = [v / scale for v in aug[col]]
        for row in range(n):
            if row != col:
                scale = aug[row][col]
                aug[row] = [v - scale * w
                            for v, w in zip(aug[row], aug[col])]
    return [row[n:] for row in aug]


def beta_continued_fraction(a, b, x):
    """Continued fraction for the incomplete beta function (Numerical Recipes)."""
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
            return h
    raise ArithmeticError("Incomplete beta calculation did not converge")


def regularized_beta(x, a, b):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                     + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * beta_continued_fraction(a, b, x) / a
    return 1.0 - front * beta_continued_fraction(b, a, 1.0 - x) / b


def two_sided_t_pvalue(statistic, degrees_freedom):
    x = degrees_freedom / (degrees_freedom + statistic * statistic)
    return regularized_beta(x, degrees_freedom / 2.0, 0.5)


def analyze(data_path):
    with open(data_path, newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        required = {"site", "treatment", "employed_m12"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("Data must contain site, treatment, and employed_m12 columns")
        rows = list(reader)
    if not rows:
        raise ValueError("Data file contains no observations")

    sites = sorted({row["site"] for row in rows})
    if any(not site for site in sites):
        raise ValueError("Site values must be present")
    # Treatment is the coefficient of interest; the first site is the reference.
    x = []
    y = []
    for row in rows:
        treatment = float(row["treatment"])
        outcome = float(row["employed_m12"])
        if treatment not in (0.0, 1.0):
            raise ValueError("treatment must be coded 0/1")
        x.append([1.0, treatment] + [float(row["site"] == site)
                                    for site in sites[1:]])
        y.append(outcome)

    xt = transpose(x)
    xtx_inv = inverse(matmul(xt, x))
    xty = [[sum(xt[i][j] * y[j] for j in range(len(y)))]
           for i in range(len(xt))]
    beta = [row[0] for row in matmul(xtx_inv, xty)]
    residuals = [y[i] - sum(x[i][j] * beta[j] for j in range(len(beta)))
                 for i in range(len(y))]
    # HC2 sandwich: (X'X)^-1 X' diag(e_i^2 / (1-h_ii)) X (X'X)^-1.
    meat = [[0.0 for _ in beta] for _ in beta]
    for i, row in enumerate(x):
        leverage = sum(row[j] * matmul(xtx_inv, [[v] for v in row])[j][0]
                       for j in range(len(beta)))
        if leverage >= 1.0:
            raise ValueError("HC2 undefined for an observation with leverage >= 1")
        weight = residuals[i] ** 2 / (1.0 - leverage)
        for j in range(len(beta)):
            for k in range(len(beta)):
                meat[j][k] += weight * row[j] * row[k]
    covariance = matmul(matmul(xtx_inv, meat), xtx_inv)
    estimate = beta[1]
    std_error = math.sqrt(covariance[1][1])
    if std_error <= 0.0:
        raise ValueError("Treatment standard error is not positive")
    df = len(y) - len(beta)
    p_value = two_sided_t_pvalue(abs(estimate / std_error), df)
    return {"estimate": estimate, "std_error": std_error, "p_value": p_value}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()
    result = analyze(args.data)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n",
                                 encoding="utf-8")


if __name__ == "__main__":
    main()
