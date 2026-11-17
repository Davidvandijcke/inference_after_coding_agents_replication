#!/usr/bin/env python3
"""Estimate the fixed, exploratory Pathways to Work finding on a new CSV.

Chosen estimand: intention-to-treat effect on usual weekly hours at 12 months,
among participants younger than 35 at random assignment (age < 35).
Model: hours_m12 ~ treatment + categorical site, including an intercept.
Inference: HC2 standard error and two-sided t test with residual degrees of
freedom. The p-value is nominal; it does not adjust for the original search.

Dependencies: numpy, pandas, scipy, statsmodels.
Usage: python score.py --data data.csv --output result.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def analyze(data_path):
    """Run only the selected subgroup, outcome and regression specification."""
    data = pd.read_csv(data_path, usecols=["age", "site", "treatment", "hours_m12"])
    if data.isna().any().any():
        raise ValueError("Required columns must be fully observed, as in the codebook.")
    for name in ("age", "treatment", "hours_m12"):
        data[name] = pd.to_numeric(data[name], errors="raise")
        if not np.isfinite(data[name].to_numpy(dtype=float)).all():
            raise ValueError(f"{name} must contain finite numeric values.")
    if not data.treatment.isin([0, 1]).all():
        raise ValueError("treatment must be coded 0 or 1.")
    subgroup = data.loc[data.age.lt(35)].copy()
    if subgroup.empty or subgroup.treatment.nunique() != 2:
        raise ValueError("The age < 35 subgroup must contain both treatment arms.")
    model = smf.ols("hours_m12 ~ treatment + C(site)", data=subgroup)
    design = model.exog
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError("Treatment and site effects must be jointly identifiable.")
    if design.shape[0] <= design.shape[1]:
        raise ValueError("The subgroup must have positive residual degrees of freedom.")
    fit = model.fit(cov_type="HC2", use_t=True)
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("The chosen regression did not yield finite inference.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Input participant CSV")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON file")
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error("Output must differ from the input data path.")
    result = analyze(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
