#!/usr/bin/env python3
"""Estimate the site-adjusted ITT effect on employment at 12 months.

Usage: python score.py --data data.csv --output result.json
Requires pandas, numpy, scipy, and statsmodels.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def chosen_analysis(data_path):
    """Run only employed_m12 ~ treatment + C(site), with HC2 t inference."""
    columns = ["employed_m12", "treatment", "site"]
    data = pd.read_csv(data_path, usecols=columns)
    if data.empty or data.isna().any().any():
        raise ValueError("Analysis columns must be nonempty and fully observed.")
    if set(data["treatment"].unique()) != {0, 1}:
        raise ValueError("Treatment must contain both 0 and 1.")
    if not data["employed_m12"].isin([0, 1]).all():
        raise ValueError("employed_m12 must be binary (0 or 1).")

    model = smf.ols("employed_m12 ~ treatment + C(site)", data=data)
    design = model.exog
    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("The treatment and site design must have full rank.")
    if len(data) <= design.shape[1]:
        raise ValueError("The model requires positive residual degrees of freedom.")
    fit = model.fit(cov_type="HC2", use_t=True)
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("The chosen model produced nonfinite statistics.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = chosen_analysis(args.data)
    args.output.write_text(json.dumps(result, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
