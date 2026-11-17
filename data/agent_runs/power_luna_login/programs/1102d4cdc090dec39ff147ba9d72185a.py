#!/usr/bin/env python3
"""Score the site-adjusted treatment effect on employment at month 12."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def _beta_continued_fraction(a, b, x):
    """Continued fraction used by the regularized incomplete beta function."""
    tiny = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 1001):
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


def _two_sided_t_p_value(t_stat, df):
    x = df / (df + t_stat * t_stat)
    return _regularized_beta(x, df / 2.0, 0.5)


def score(data_path):
    data = pd.read_csv(data_path)
    required = {"site", "treatment", "employed_m12"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))
    if data[list(required)].isnull().any().any():
        raise ValueError("Required analysis columns must not contain missing values")

    site = pd.get_dummies(data["site"], drop_first=True, dtype=float).to_numpy()
    treatment = data["treatment"].to_numpy(dtype=float)
    outcome = data["employed_m12"].to_numpy(dtype=float)
    design = np.column_stack((np.ones(len(data)), treatment, site))

    xtx_inv = np.linalg.inv(design.T @ design)
    coefficients = xtx_inv @ design.T @ outcome
    residuals = outcome - design @ coefficients
    leverage = np.sum((design @ xtx_inv) * design, axis=1)
    hc2_meat = design.T @ (design * (residuals**2 / (1.0 - leverage))[:, None])
    covariance = xtx_inv @ hc2_meat @ xtx_inv

    estimate = float(coefficients[1])
    std_error = float(math.sqrt(covariance[1, 1]))
    df = len(data) - design.shape[1]
    p_value = _two_sided_t_p_value(estimate / std_error, df)
    return {"estimate": estimate, "std_error": std_error, "p_value": p_value}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()
    result = score(args.data)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
