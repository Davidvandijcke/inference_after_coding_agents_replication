#!/usr/bin/env python3
"""Run the fixed Pathways to Work analysis (requires NumPy and SciPy).

Population: months_unemployed_baseline >= 6.
Outcome: earnings_m6, in dollars earned in the previous month.
Model: OLS on an intercept, treatment, and site indicators; no other covariates.
Inference: HC2 standard error and two-sided Student t test, df = n - rank(X).

Usage: python score.py --data data.csv --output result.json
This file does not search outcomes, subgroups, cutoffs, or specifications.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import t


def chosen_analysis(data_path):
    required = {"months_unemployed_baseline", "earnings_m6", "treatment", "site"}
    rows = []
    with open(data_path, newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError("Missing columns: " + ", ".join(sorted(missing)))
        for row in reader:
            duration = float(row["months_unemployed_baseline"])
            if not np.isfinite(duration):
                raise ValueError("Unemployment duration must be finite.")
            if duration >= 6:
                treatment = float(row["treatment"])
                earnings = float(row["earnings_m6"])
                site = row["site"].strip()
                if treatment not in (0.0, 1.0):
                    raise ValueError("Treatment must be 0 or 1.")
                if not np.isfinite(earnings) or not site:
                    raise ValueError("Selected outcomes and sites must be observed.")
                rows.append((treatment, earnings, site))
    if not rows:
        raise ValueError("No participants meet the fixed subgroup criterion.")
    sites = sorted({row[2] for row in rows})
    x = np.asarray([
        [1.0, treatment] + [float(site == level) for level in sites[1:]]
        for treatment, _, site in rows
    ])
    y = np.asarray([row[1] for row in rows])
    n, p = x.shape
    if n <= p or np.linalg.matrix_rank(x) != p:
        raise ValueError("The selected regression lacks residual degrees of freedom or full rank.")

    # QR gives the OLS coefficients and HC2 sandwich without forming X'X.
    q, r = np.linalg.qr(x, mode="reduced")
    x_pinv = np.linalg.solve(r, q.T)
    coefficients = x_pinv @ y
    residuals = y - x @ coefficients
    leverage = np.sum(q * q, axis=1)
    if np.any(1.0 - leverage <= np.finfo(float).eps * 10):
        raise ValueError("HC2 is undefined for an observation with unit leverage.")
    hc2_variances = residuals**2 / (1.0 - leverage)
    se = float(np.sqrt(np.sum(x_pinv[1]**2 * hc2_variances)))
    estimate = float(coefficients[1])
    if not np.isfinite(se) or se <= 0:
        raise ValueError("The selected test requires a finite, positive standard error.")
    p_value = float(2.0 * t.sf(abs(estimate / se), df=n - p))
    return {"estimate": estimate, "std_error": se, "p_value": p_value}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error("The output path must differ from the input data path.")
    try:
        result = chosen_analysis(args.data)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
