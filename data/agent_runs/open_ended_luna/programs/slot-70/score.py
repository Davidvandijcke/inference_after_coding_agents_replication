#!/usr/bin/env python3
"""Score the exploratory 6-month earnings analysis among parents.

Analysis: OLS of earnings_m6 on treatment and site indicators, restricted to
participants with parent == 1. The treatment coefficient uses HC2 standard
errors and a two-sided t test with residual degrees of freedom.

Uses only the Python standard library.
"""

import argparse
import csv
import json
import math
from pathlib import Path


def matrix_inverse(matrix):
    """Invert a nonsingular square matrix with Gauss-Jordan elimination."""
    n = len(matrix)
    augmented = [
        [float(value) for value in row]
        + [1.0 if i == j else 0.0 for j in range(n)]
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
            if scale:
                augmented[row] = [
                    left - scale * right
                    for left, right in zip(augmented[row], augmented[col])
                ]

    return [row[n:] for row in augmented]


def matrix_vector(matrix, vector):
    return [sum(value * vector[j] for j, value in enumerate(row)) for row in matrix]


def regularized_beta(x, a, b):
    """Regularized incomplete beta function, evaluated by a continued fraction."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    def continued_fraction(aa, bb, xx):
        qab = aa + bb
        qap = aa + 1.0
        qam = aa - 1.0
        tiny = 1e-300
        c = 1.0
        d = 1.0 - qab * xx / qap
        if abs(d) < tiny:
            d = tiny
        d = 1.0 / d
        result = d

        for m in range(1, 10001):
            m2 = 2 * m
            numerator = m * (bb - m) * xx / ((qam + m2) * (aa + m2))
            d = 1.0 + numerator * d
            if abs(d) < tiny:
                d = tiny
            c = 1.0 + numerator / c
            if abs(c) < tiny:
                c = tiny
            d = 1.0 / d
            result *= d * c

            numerator = -(aa + m) * (qab + m) * xx / (
                (aa + m2) * (qap + m2)
            )
            d = 1.0 + numerator * d
            if abs(d) < tiny:
                d = tiny
            c = 1.0 + numerator / c
            if abs(c) < tiny:
                c = tiny
            d = 1.0 / d
            delta = d * c
            result *= delta
            if abs(delta - 1.0) < 3e-14:
                return result

        raise ArithmeticError("incomplete beta continued fraction did not converge")

    log_factor = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    factor = math.exp(log_factor)
    if x < (a + 1.0) / (a + b + 2.0):
        return factor * continued_fraction(a, b, x) / a
    return 1.0 - factor * continued_fraction(b, a, 1.0 - x) / b


def two_sided_t_p_value(statistic, degrees_freedom):
    if degrees_freedom <= 0:
        raise ValueError("not enough observations to estimate residual degrees of freedom")
    if statistic == 0.0:
        return 1.0
    x = degrees_freedom / (degrees_freedom + statistic * statistic)
    return regularized_beta(x, degrees_freedom / 2.0, 0.5)


def load_parent_rows(data_path):
    required = {"site", "treatment", "parent", "earnings_m6"}
    rows = []
    with Path(data_path).open("r", newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError("missing required columns: " + ", ".join(sorted(missing)))
        for row in reader:
            if int(row["parent"]) == 1:
                rows.append(
                    {
                        "site": row["site"],
                        "treatment": float(row["treatment"]),
                        "earnings": float(row["earnings_m6"]),
                    }
                )
    if not rows:
        raise ValueError("data contains no participants with parent == 1")
    return rows


def score(data_path):
    rows = load_parent_rows(data_path)
    sites = sorted({row["site"] for row in rows})
    reference_site = sites[0]

    # Intercept, treatment, and site indicators (first site is the reference).
    design = []
    outcomes = []
    for row in rows:
        design.append(
            [
                1.0,
                row["treatment"],
                *[1.0 if row["site"] == site else 0.0 for site in sites if site != reference_site],
            ]
        )
        outcomes.append(row["earnings"])

    n = len(design)
    k = len(design[0])
    if n <= k:
        raise ValueError("not enough parent subgroup observations for this model")

    xtx = [[0.0] * k for _ in range(k)]
    xty = [0.0] * k
    for x, y in zip(design, outcomes):
        for j in range(k):
            xty[j] += x[j] * y
            for ell in range(k):
                xtx[j][ell] += x[j] * x[ell]

    bread = matrix_inverse(xtx)
    coefficients = matrix_vector(bread, xty)
    residuals = [y - sum(x[j] * coefficients[j] for j in range(k))
                 for x, y in zip(design, outcomes)]

    # HC2 meat: sum_i x_i x_i' * e_i^2 / (1 - h_i).
    meat = [[0.0] * k for _ in range(k)]
    for x, residual in zip(design, residuals):
        leverage = sum(
            x[j] * bread[j][ell] * x[ell]
            for j in range(k)
            for ell in range(k)
        )
        adjustment = 1.0 - leverage
        if adjustment <= 0.0:
            raise ValueError("HC2 leverage adjustment is not positive")
        weight = residual * residual / adjustment
        for j in range(k):
            for ell in range(k):
                meat[j][ell] += weight * x[j] * x[ell]

    temporary = [
        [sum(bread[i][j] * meat[j][ell] for j in range(k)) for ell in range(k)]
        for i in range(k)
    ]
    covariance = [
        [sum(temporary[i][j] * bread[j][ell] for j in range(k)) for ell in range(k)]
        for i in range(k)
    ]

    estimate = coefficients[1]
    variance = covariance[1][1]
    if variance <= 0.0:
        raise ValueError("estimated treatment variance is not positive")
    std_error = math.sqrt(variance)
    p_value = two_sided_t_p_value(estimate / std_error, n - k)
    return {"estimate": estimate, "std_error": std_error, "p_value": p_value}


def main():
    parser = argparse.ArgumentParser(
        description="Score the site-adjusted 6-month earnings effect among parents."
    )
    parser.add_argument("--data", required=True, help="input CSV path")
    parser.add_argument("--output", required=True, help="output JSON path")
    args = parser.parse_args()

    result = score(args.data)
    with Path(args.output).open("w", encoding="utf-8") as destination:
        json.dump(result, destination, indent=2)
        destination.write("\n")


if __name__ == "__main__":
    main()
