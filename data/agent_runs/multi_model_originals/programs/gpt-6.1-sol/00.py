#!/usr/bin/env python3
"""Frozen analysis: full-sample effect on self-rated health at 12 months.

Model: health_m12 ~ treatment + C(site), with an intercept, HC2 covariance,
and a two-sided residual-df t test of treatment = 0. No search or adjustment
for the original exploratory selection is performed here.

Requires numpy, pandas, and statsmodels (which depends on scipy).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def score(data_path):
    data = pd.read_csv(data_path)
    required = ["health_m12", "treatment", "site"]
    missing = set(required) - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    data = data.loc[:, required].copy()
    if data.isna().any().any():
        raise ValueError("The chosen analysis requires complete observations.")
    for column in ["health_m12", "treatment"]:
        data[column] = pd.to_numeric(data[column], errors="raise")
        if not np.isfinite(data[column]).all():
            raise ValueError(f"Nonfinite values in {column}.")
    if set(data.treatment.unique()) != {0, 1}:
        raise ValueError("Both treatment arms, coded 0 and 1, are required.")

    model = smf.ols("health_m12 ~ treatment + C(site)", data=data,
                    missing="raise")
    design = model.exog
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError("The treatment effect is not identified given site.")
    if design.shape[0] <= design.shape[1]:
        raise ValueError("Positive residual degrees of freedom are required.")
    fit = model.fit(cov_type="HC2", use_t=True)
    return {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error("The output path must differ from the input data path.")
    # Serialize before opening the output so undefined inference fails cleanly.
    payload = json.dumps(score(args.data), indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
