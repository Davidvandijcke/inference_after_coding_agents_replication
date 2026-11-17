#!/usr/bin/env python3
"""Score the selected Pathways to Work subgroup analysis.

Selected test: effect of treatment on 12-month financial security among
participants unemployed at baseline, adjusting for site and using HC2 robust
standard errors with a two-sided t-test.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = {"treatment", "site", "employed_baseline", "fin_security_m12"}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    sample = data.loc[data["employed_baseline"] == 0].copy()
    if sample.empty:
        raise ValueError("No participants unemployed at baseline")

    sites = sorted(sample["site"].dropna().unique())
    # Intercept, treatment, and site indicators (first observed site is reference).
    columns = [
        np.ones(len(sample), dtype=float),
        sample["treatment"].to_numpy(dtype=float),
    ]
    columns.extend(
        (sample["site"].to_numpy() == site).astype(float) for site in sites[1:]
    )
    design = np.column_stack(columns)
    outcome = sample["fin_security_m12"].to_numpy(dtype=float)

    xtx_inv = np.linalg.inv(design.T @ design)
    coefficients = xtx_inv @ design.T @ outcome
    residuals = outcome - design @ coefficients
    leverage = np.einsum("ij,jk,ik->i", design, xtx_inv, design)
    hc2_meat = design.T @ ((residuals**2 / (1.0 - leverage))[:, None] * design)
    covariance = xtx_inv @ hc2_meat @ xtx_inv

    estimate = float(coefficients[1])
    std_error = float(np.sqrt(covariance[1, 1]))
    degrees_freedom = len(sample) - design.shape[1]
    statistic = estimate / std_error
    p_value = float(2.0 * student_t.sf(abs(statistic), degrees_freedom))

    result = {
        "estimate": estimate,
        "std_error": std_error,
        "p_value": p_value,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
