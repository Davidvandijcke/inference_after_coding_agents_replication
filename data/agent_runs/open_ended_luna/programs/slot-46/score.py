#!/usr/bin/env python3
"""Score the six-month earnings interaction by baseline employment status.

Fits OLS to earnings_m6 with treatment, baseline employment, their interaction,
site indicators, and baseline earnings. Reports the HC2 standard error and the
two-sided t-test for the treatment-by-baseline-employment interaction.
Only Python's standard library is required.
"""

import argparse
import csv
import json
import math
from pathlib import Path


def invert(matrix):
    """Invert a small square matrix with Gauss-Jordan elimination."""
    size = len(matrix)
    augmented = [
        [float(value) for value in row]
        + [1.0 if i == j else 0.0 for j in range(size)]
        for i, row in enumerate(matrix)
    ]
    for col in range(size):
        pivot = max(range(col, size), key=lambda row: abs(augmented[row][col]))
        if abs(augmented[pivot][col]) < 1e-12:
            raise ValueError("Regression design matrix is rank deficient")
        augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        divisor = augmented[col][col]
        augmented[col] = [value / divisor for value in augmented[col]]
        for row in range(size):
            if row == col:
                continue
            factor = augmented[row][col]
            if factor:
                augmented[row] = [
                    value - factor * pivot_value
                    for value, pivot_value in zip(augmented[row], augmented[col])
                ]
    return [row[size:] for row in augmented]


def mat_vec(matrix, vector):
    return [sum(a * b for a, b in zip(row, vector)) for row in matrix]


def beta_continued_fraction(a, b, x):
    """Continued fraction used to evaluate the regularized incomplete beta."""
    max_iterations = 300
    epsilon = 3e-14
    tiny = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    result = d

    for m in range(1, max_iterations + 1):
        m2 = 2 * m
        term = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + term * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + term / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        result *= d * c

        term = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + term * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + term / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) < epsilon:
            return result

    raise ArithmeticError("Incomplete beta calculation did not converge")


def regularized_beta(x, a, b):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    front = math.exp(log_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * beta_continued_fraction(a, b, x) / a
    return 1.0 - front * beta_continued_fraction(b, a, 1.0 - x) / b


def two_sided_t_p_value(t_statistic, degrees_freedom):
    x = degrees_freedom / (degrees_freedom + t_statistic * t_statistic)
    return regularized_beta(x, degrees_freedom / 2.0, 0.5)


def fit_interaction(rows):
    site_values = sorted({row["site"] for row in rows})
    if len(site_values) < 2:
        raise ValueError("At least two sites are needed for site-adjusted estimation")

    # The reference site is omitted. Scaling baseline earnings improves the
    # conditioning of X'X without changing the treatment interaction estimate.
    columns = ["intercept", "treatment", "employed", "interaction", "earnings_k"]
    columns.extend(f"site_{site}" for site in site_values[1:])
    design = []
    outcome = []
    for row in rows:
        treatment = row["treatment"]
        employed = row["employed_baseline"]
        design.append(
            [
                1.0,
                treatment,
                employed,
                treatment * employed,
                row["earnings_baseline"] / 1000.0,
            ]
            + [1.0 if row["site"] == site else 0.0 for site in site_values[1:]]
        )
        outcome.append(row["earnings_m6"])

    n = len(design)
    k = len(columns)
    if n <= k:
        raise ValueError("Not enough observations for the regression")

    xtx = [[0.0] * k for _ in range(k)]
    xty = [0.0] * k
    for vector, y in zip(design, outcome):
        for i in range(k):
            xty[i] += vector[i] * y
            for j in range(k):
                xtx[i][j] += vector[i] * vector[j]

    bread = invert(xtx)
    coefficients = mat_vec(bread, xty)
    residuals = [
        y - sum(a * b for a, b in zip(vector, coefficients))
        for vector, y in zip(design, outcome)
    ]

    meat = [[0.0] * k for _ in range(k)]
    for vector, residual in zip(design, residuals):
        leverage = sum(
            vector[i] * bread[i][j] * vector[j]
            for i in range(k)
            for j in range(k)
        )
        correction = 1.0 - leverage
        if correction <= 1e-12:
            raise ValueError("HC2 variance is undefined for an observation with leverage 1")
        weight = residual * residual / correction
        for i in range(k):
            for j in range(k):
                meat[i][j] += weight * vector[i] * vector[j]

    covariance = [
        [
            sum(
                bread[i][a] * meat[a][b] * bread[b][j]
                for a in range(k)
                for b in range(k)
            )
            for j in range(k)
        ]
        for i in range(k)
    ]

    interaction_index = columns.index("interaction")
    estimate = coefficients[interaction_index]
    std_error = math.sqrt(covariance[interaction_index][interaction_index])
    if std_error <= 0.0:
        raise ValueError("Estimated standard error is not positive")
    p_value = two_sided_t_p_value(estimate / std_error, n - k)
    return {"estimate": estimate, "std_error": std_error, "p_value": p_value}


def read_data(path):
    required = {
        "site",
        "treatment",
        "employed_baseline",
        "earnings_baseline",
        "earnings_m6",
    }
    parsed = []
    with open(path, "r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))
        for line_number, raw in enumerate(reader, start=2):
            try:
                treatment = float(raw["treatment"])
                employed = float(raw["employed_baseline"])
                earnings_baseline = float(raw["earnings_baseline"])
                earnings_m6 = float(raw["earnings_m6"])
                site = raw["site"]
                if not site:
                    raise ValueError("empty site")
                numeric_values = (treatment, employed, earnings_baseline, earnings_m6)
                if not all(math.isfinite(value) for value in numeric_values):
                    raise ValueError("non-finite value")
                if treatment not in (0.0, 1.0) or employed not in (0.0, 1.0):
                    raise ValueError("treatment and employed_baseline must be 0 or 1")
                parsed.append(
                    {
                        "site": site,
                        "treatment": treatment,
                        "employed_baseline": employed,
                        "earnings_baseline": earnings_baseline,
                        "earnings_m6": earnings_m6,
                    }
                )
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid value on CSV line {line_number}: {error}") from error

    if not parsed:
        raise ValueError("Input CSV has no observations")
    return parsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV dataset")
    parser.add_argument("--output", required=True, help="Path for JSON result")
    args = parser.parse_args()

    result = fit_interaction(read_data(args.data))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
