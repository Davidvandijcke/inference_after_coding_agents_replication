#!/usr/bin/env python3
"""Score the 12-month employment treatment effect with site adjustment and HC2 SEs."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def _beta_continued_fraction(a, b, x):
    """Continued fraction used by the regularized incomplete beta function."""
    max_iter = 300
    eps = 3.0e-14
    tiny = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
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


def _regularized_incomplete_beta(x, a, b):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    factor = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return factor * _beta_continued_fraction(a, b, x) / a
    return 1.0 - factor * _beta_continued_fraction(b, a, 1.0 - x) / b


def _two_sided_t_pvalue(t_statistic, degrees_freedom):
    x = degrees_freedom / (degrees_freedom + t_statistic * t_statistic)
    return _regularized_incomplete_beta(x, degrees_freedom / 2.0, 0.5)


def estimate(data_path):
    groups = defaultdict(list)
    with open(data_path, newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        required = {"site", "treatment", "employed_m12"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Input must contain site, treatment, and employed_m12 columns")
        for row in reader:
            site = row["site"]
            treatment = float(row["treatment"])
            outcome = float(row["employed_m12"])
            groups[site].append((treatment, outcome))

    n = sum(len(rows) for rows in groups.values())
    if n == 0 or len(groups) < 2:
        raise ValueError("Input must contain observations from at least two sites")

    site_stats = {}
    sxx = 0.0
    numerator = 0.0
    for site, rows in groups.items():
        n_site = len(rows)
        mean_t = sum(t for t, _ in rows) / n_site
        mean_y = sum(y for _, y in rows) / n_site
        site_stats[site] = (n_site, mean_t, mean_y)
        for treatment, outcome in rows:
            centered_t = treatment - mean_t
            sxx += centered_t * centered_t
            numerator += centered_t * (outcome - mean_y)

    if sxx <= 0.0:
        raise ValueError("Treatment must vary within sites")
    effect = numerator / sxx

    hc2_variance = 0.0
    for site, rows in groups.items():
        n_site, mean_t, mean_y = site_stats[site]
        for treatment, outcome in rows:
            centered_t = treatment - mean_t
            residual = (outcome - mean_y) - effect * centered_t
            leverage = 1.0 / n_site + centered_t * centered_t / sxx
            hc2_variance += (
                (centered_t / sxx) ** 2 * residual * residual / (1.0 - leverage)
            )

    standard_error = math.sqrt(hc2_variance)
    degrees_freedom = n - len(groups) - 1
    if degrees_freedom <= 0 or standard_error <= 0.0:
        raise ValueError("Insufficient data to estimate inference")
    p_value = _two_sided_t_pvalue(effect / standard_error, degrees_freedom)
    return {
        "estimate": effect,
        "std_error": standard_error,
        "p_value": p_value,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Estimate the site-adjusted effect on employment at month 12."
    )
    parser.add_argument("--data", required=True, help="CSV file with the evaluation columns")
    parser.add_argument("--output", required=True, help="Path for the JSON result")
    args = parser.parse_args()

    result = estimate(args.data)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
