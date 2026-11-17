#!/usr/bin/env python3
"""Run the fixed Pathways to Work analysis selected in the exploratory report.

Estimand: assignment effect on 12-month employment among participants whose
baseline education is 1 or 2 (high school/GED or less). Regress employed_m12 on
treatment and categorical site indicators, with an intercept and no additional
covariates. Use HC2 standard errors and a two-sided Student t test with OLS
residual degrees of freedom. Estimates are probability differences (0.01 = 1 pp).

Usage: python score.py --data data.csv --output result.json
Dependencies: numpy, pandas, scipy, statsmodels.
No outcome, subgroup, cutoff, or model selection is performed here.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def analyze(data_path):
    required = ["education", "site", "treatment", "employed_m12"]
    data = pd.read_csv(data_path, usecols=required)
    for column in required:
        data[column] = pd.to_numeric(data[column], errors="raise")
    if not np.isfinite(data[required].to_numpy()).all():
        raise ValueError("Required columns must contain only finite, observed values.")
    for column, allowed in {
        "education": [1, 2, 3, 4],
        "site": [1, 2, 3, 4],
        "treatment": [0, 1],
        "employed_m12": [0, 1],
    }.items():
        if not data[column].isin(allowed).all():
            raise ValueError(f"Invalid code in {column}; expected {allowed}.")

    sample = data.loc[data["education"].isin([1, 2])].copy()
    if sample["treatment"].nunique() != 2:
        raise ValueError("The selected subgroup must contain both treatment arms.")
    model = smf.ols("employed_m12 ~ treatment + C(site)", data=sample)
    design = model.exog
    if len(sample) <= design.shape[1] or np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("The selected regression needs full rank and positive residual degrees of freedom.")
    leverage = np.sum(design * np.linalg.pinv(design).T, axis=1)
    if np.any(leverage >= 1 - 1e-12):
        raise ValueError("HC2 inference is undefined for an observation with unit leverage.")
    result = model.fit(cov_type="HC2", use_t=True)
    output = {
        "estimate": float(result.params["treatment"]),
        "std_error": float(result.bse["treatment"]),
        "p_value": float(result.pvalues["treatment"]),
    }
    if not all(np.isfinite(value) for value in output.values()):
        raise ValueError("The selected regression did not produce finite inference.")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error("--output must differ from --data; input observations are read-only.")
    output = analyze(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
