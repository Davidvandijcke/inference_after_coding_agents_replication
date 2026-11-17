#!/usr/bin/env python3
"""Site-adjusted HC2 test of the 12-month employment treatment effect."""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def beta_continued_fraction(a, b, x):
    """Continued fraction for the incomplete beta function (Numerical Recipes)."""
    max_iter, eps, tiny = 300, 3e-14, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
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


def regularized_beta(x, a, b):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Input CSV dataset')
    parser.add_argument('--output', required=True, help='Output JSON path')
    args = parser.parse_args()

    with open(args.data, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError('Input data has no rows')
    required = {'treatment', 'site', 'employed_m12'}
    if not required.issubset(rows[0]):
        raise ValueError('Input CSV must contain treatment, site, and employed_m12')

    # Intercept, treatment, and site indicators (lowest site is the reference).
    sites = sorted({int(row['site']) for row in rows})
    x = np.array([
        [1.0, float(row['treatment'])]
        + [float(int(row['site']) == site) for site in sites[1:]]
        for row in rows
    ])
    y = np.array([float(row['employed_m12']) for row in rows])
    n, k = x.shape
    if n <= k or np.linalg.matrix_rank(x) != k:
        raise ValueError('Design matrix is not full rank or has insufficient rows')

    xtx_inv = np.linalg.inv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    residual = y - x @ beta
    leverage = np.einsum('ij,jk,ik->i', x, xtx_inv, x)
    weighted_x = x * (residual**2 / (1.0 - leverage))[:, None]
    covariance = xtx_inv @ (x.T @ weighted_x) @ xtx_inv
    estimate = float(beta[1])
    std_error = float(math.sqrt(covariance[1, 1]))
    df = n - k
    t_stat = estimate / std_error
    x_beta = df / (df + t_stat * t_stat)
    p_value = regularized_beta(x_beta, df / 2.0, 0.5)

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump({'estimate': estimate, 'std_error': std_error, 'p_value': p_value}, f, indent=2)
        f.write('\n')


if __name__ == '__main__':
    main()
