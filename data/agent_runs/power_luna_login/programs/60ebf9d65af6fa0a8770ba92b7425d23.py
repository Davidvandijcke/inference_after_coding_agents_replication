#!/usr/bin/env python3
"""Score the site-adjusted Pathways to Work effect on employment at 12 months.

Usage: python score.py --data <path> --output <path>

Fits OLS of employed_m12 on treatment and site indicators, then reports the
treatment coefficient, HC2 standard error, and two-sided t-test p-value.
"""

import argparse
import csv
import json
import math
from collections import defaultdict


def _beta_continued_fraction(a, b, x):
    """Continued fraction for the incomplete beta function (Numerical Recipes)."""
    max_iterations = 200
    epsilon = 3.0e-14
    floor = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < floor:
        d = floor
    d = 1.0 / d
    h = d
    for m in range(1, max_iterations + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < floor:
            d = floor
        c = 1.0 + aa / c
        if abs(c) < floor:
            c = floor
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < floor:
            d = floor
        c = 1.0 + aa / c
        if abs(c) < floor:
            c = floor
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < epsilon:
            return h
    raise ArithmeticError("incomplete beta calculation did not converge")


def _regularized_beta(x, a, b):
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
        return front * _beta_continued_fraction(a, b, x) / a
    return 1.0 - front * _beta_continued_fraction(b, a, 1.0 - x) / b


def _two_sided_t_p_value(statistic, degrees_freedom):
    x = degrees_freedom / (degrees_freedom + statistic * statistic)
    return _regularized_beta(x, degrees_freedom / 2.0, 0.5)


def analyze(path):
    sites = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        required = {"site", "treatment", "employed_m12"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("data must include site, treatment, and employed_m12 columns")
        for row in reader:
            site = row["site"]
            treatment = float(row["treatment"])
            outcome = float(row["employed_m12"])
            if treatment not in (0.0, 1.0):
                raise ValueError("treatment must be coded 0 or 1")
            sites[site].append((treatment, outcome))

    n = sum(len(rows) for rows in sites.values())
    if len(sites) < 1 or n == 0:
        raise ValueError("data contains no participants")

    # Frisch-Waugh-Lovell: remove site means from treatment and outcome.
    centered = []
    treatment_ss = 0.0
    for rows in sites.values():
        count = len(rows)
        mean_t = sum(t for t, _ in rows) / count
        mean_y = sum(y for _, y in rows) / count
        current = [(t - mean_t, y - mean_y) for t, y in rows]
        centered.append((count, current))
        treatment_ss += sum(tw * tw for tw, _ in current)
    if treatment_ss <= 0.0:
        raise ValueError("treatment has no within-site variation")

    treatment_effect = sum(
        tw * yw for _, rows in centered for tw, yw in rows
    ) / treatment_ss
    variance_numerator = 0.0
    for count, rows in centered:
        for tw, yw in rows:
            residual = yw - treatment_effect * tw
            leverage = 1.0 / count + tw * tw / treatment_ss
            if leverage >= 1.0:
                raise ValueError("HC2 is undefined for a leverage-one observation")
            variance_numerator += tw * tw * residual * residual / (1.0 - leverage)

    standard_error = math.sqrt(variance_numerator) / treatment_ss
    degrees_freedom = n - len(sites) - 1
    if standard_error <= 0.0 or degrees_freedom <= 0:
        raise ValueError("cannot calculate the requested t-test")
    statistic = treatment_effect / standard_error
    p_value = _two_sided_t_p_value(statistic, degrees_freedom)
    return {
        "estimate": treatment_effect,
        "std_error": standard_error,
        "p_value": p_value,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="input CSV path")
    parser.add_argument("--output", required=True, help="output JSON path")
    args = parser.parse_args()
    result = analyze(args.data)
    with open(args.output, "w", encoding="utf-8") as destination:
        json.dump(result, destination, indent=2)
        destination.write("\n")


if __name__ == "__main__":
    main()
