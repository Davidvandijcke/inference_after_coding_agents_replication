#!/usr/bin/env python3
"""Score the site-adjusted intent-to-treat effect on 12-month employment.

Usage: python score.py --data <path> --output <path>

Fits employment_m12 on treatment and site indicators, calculates an HC2
heteroskedasticity-robust standard error, and reports a two-sided t-test.
Only Python's standard library is required.
"""

import argparse
import csv
import json
import math
from pathlib import Path


def solve(matrix, vector):
    """Solve a small square linear system by Gauss-Jordan elimination."""
    n = len(vector)
    aug = [list(map(float, matrix[i])) + [float(vector[i])] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(aug[row][col]))
        if abs(aug[pivot][col]) < 1e-12:
            raise ValueError("Regression design matrix is singular")
        aug[col], aug[pivot] = aug[pivot], aug[col]
        scale = aug[col][col]
        aug[col] = [value / scale for value in aug[col]]
        for row in range(n):
            if row == col:
                continue
            scale = aug[row][col]
            if scale:
                aug[row] = [a - scale * b for a, b in zip(aug[row], aug[col])]
    return [aug[i][-1] for i in range(n)]


def inverse(matrix):
    """Invert a small matrix by solving against each unit vector."""
    n = len(matrix)
    columns = [solve(matrix, [float(i == j) for i in range(n)]) for j in range(n)]
    return [[columns[j][i] for j in range(n)] for i in range(n)]


def beta_continued_fraction(a, b, x):
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    tiny = 1e-300
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 301):
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
            break
    return h


def regularized_incomplete_beta(x, a, b):
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


def two_sided_t_pvalue(t_statistic, degrees_freedom):
    x = degrees_freedom / (degrees_freedom + t_statistic * t_statistic)
    return regularized_incomplete_beta(x, degrees_freedom / 2.0, 0.5)


def score(data_path):
    with open(data_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"site", "treatment", "employed_m12"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("Data must include site, treatment, and employed_m12 columns")
        rows = list(reader)

    if not rows:
        raise ValueError("Data file has no observations")
    sites = sorted({row["site"] for row in rows})
    if len(sites) < 1:
        raise ValueError("No site values found")
    site_reference = sites[0]

    # The design is intercept, all but one site indicator, and treatment.
    design = []
    outcomes = []
    for row in rows:
        site = row["site"]
        treatment = float(row["treatment"])
        outcome = float(row["employed_m12"])
        if treatment not in (0.0, 1.0) or outcome not in (0.0, 1.0):
            raise ValueError("treatment and employed_m12 must contain only 0 or 1")
        design.append([1.0] + [float(site == level) for level in sites if level != site_reference] + [treatment])
        outcomes.append(outcome)

    n = len(rows)
    k = len(design[0])
    if n <= k:
        raise ValueError("Not enough observations for the requested regression")
    xtx = [[sum(x[i] * x[j] for x in design) for j in range(k)] for i in range(k)]
    xty = [sum(x[i] * y for x, y in zip(design, outcomes)) for i in range(k)]
    xtx_inverse = inverse(xtx)
    coefficients = [sum(xtx_inverse[i][j] * xty[j] for j in range(k)) for i in range(k)]
    residuals = [y - sum(a * b for a, b in zip(x, coefficients)) for x, y in zip(design, outcomes)]

    # HC2 sandwich covariance: (X'X)^-1 X' diag(e_i^2 / (1-h_i)) X (X'X)^-1.
    meat = [[0.0] * k for _ in range(k)]
    for x, residual in zip(design, residuals):
        leverage = sum(x[i] * xtx_inverse[i][j] * x[j] for i in range(k) for j in range(k))
        if leverage >= 1.0:
            raise ValueError("HC2 is undefined for an observation with leverage 1")
        weight = residual * residual / (1.0 - leverage)
        for i in range(k):
            for j in range(k):
                meat[i][j] += weight * x[i] * x[j]
    covariance = [
        [sum(xtx_inverse[i][a] * meat[a][b] * xtx_inverse[b][j]
             for a in range(k) for b in range(k)) for j in range(k)]
        for i in range(k)
    ]
    treatment_index = k - 1
    estimate = coefficients[treatment_index]
    std_error = math.sqrt(covariance[treatment_index][treatment_index])
    if std_error == 0.0:
        p_value = 0.0 if estimate != 0.0 else 1.0
    else:
        p_value = two_sided_t_pvalue(estimate / std_error, n - k)
    return {"estimate": estimate, "std_error": std_error, "p_value": p_value}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV dataset")
    parser.add_argument("--output", required=True, help="Output JSON file")
    args = parser.parse_args()
    result = score(args.data)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
