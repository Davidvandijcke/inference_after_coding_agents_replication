#!/usr/bin/env python3
"""Estimate the fixed, full-sample 12-month employment intention-to-treat effect.

Usage: python score.py --data data.csv --output result.json
Requires numpy, pandas, scipy, and statsmodels.
The estimate and standard error are in probability units (0.01 = 1 percentage point).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def analyze(data_path):
    """Run only employed_m12 ~ treatment + C(site), with HC2 and a t test."""
    data = pd.read_csv(data_path, usecols=["employed_m12", "treatment", "site"])
    if data.empty or data.isna().any().any():
        raise ValueError("The required analysis columns must be nonempty and complete.")
    if set(data["treatment"].unique()) != {0, 1}:
        raise ValueError("treatment must contain both 0 and 1.")
    if not data["employed_m12"].isin([0, 1]).all():
        raise ValueError("employed_m12 must contain only 0 and 1.")
    if not data["site"].isin([1, 2, 3, 4]).all():
        raise ValueError("site must be coded 1, 2, 3, or 4.")

    model = smf.ols("employed_m12 ~ treatment + C(site)", data=data, missing="raise")
    design = model.exog
    if len(data) <= design.shape[1] or np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("Insufficient independent observations to fit the chosen model.")
    result = model.fit(cov_type="HC2", use_t=True)
    values = {
        "estimate": float(result.params["treatment"]),
        "std_error": float(result.bse["treatment"]),
        "p_value": float(result.pvalues["treatment"]),
    }
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError("The chosen model did not produce finite inference statistics.")
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error("--output must differ from --data.")
    result = analyze(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
