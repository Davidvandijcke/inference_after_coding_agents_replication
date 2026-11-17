#!/usr/bin/env python3
"""Run the fixed Pathways to Work analysis, without exploratory selection.

Estimand: intention-to-treat effect on employed_m12 among participants with
months_unemployed_baseline >= 6, in probability units (not percentage points).
Model: OLS, treatment plus an intercept and categorical site indicators.
Inference: HC2 standard error, two-sided t test with OLS residual degrees of
freedom. The reported p-value is nominal, without multiplicity adjustment.

Requires pandas, numpy, scipy, and statsmodels.
Usage: python score.py --data data.csv --output result.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def analyze(data_path):
    data = pd.read_csv(data_path)
    required = ["site", "treatment", "months_unemployed_baseline", "employed_m12"]
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if data[required].isna().any().any():
        raise ValueError("Required analysis columns must be fully observed.")
    numeric = ["treatment", "months_unemployed_baseline", "employed_m12"]
    if not np.isfinite(data[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Analysis variables must be finite numeric values.")
    for column in ["treatment", "employed_m12"]:
        if not data[column].isin([0, 1]).all():
            raise ValueError(f"{column} must be coded 0 or 1.")

    sample = data.loc[data["months_unemployed_baseline"] >= 6].copy()
    if sample["treatment"].nunique() != 2:
        raise ValueError("The fixed subgroup must contain both assignment arms.")
    model = smf.ols("employed_m12 ~ treatment + C(site)", data=sample)
    design = model.exog
    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("Treatment and site effects are not separately identified.")
    if design.shape[0] <= design.shape[1]:
        raise ValueError("Insufficient residual degrees of freedom.")
    fit = model.fit(cov_type="HC2", use_t=True)
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    if not np.isfinite(list(result.values())).all() or result["std_error"] <= 0:
        raise ValueError("The fixed analysis did not yield valid inference.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error("Output must be separate from the input data.")
    result = analyze(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
