#!/usr/bin/env python3
"""Estimate the effect of assignment on six-month training enrollment.

Usage: python score.py --data data.csv --output result.json
Requires pandas and statsmodels. The reported p-value is the nominal,
two-sided HC2 t-test from OLS with site fixed effects.
"""

import argparse
import json
import math
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if args.data.resolve() == args.output.resolve():
        parser.error("The output path must differ from the input data path.")

    data = pd.read_csv(args.data)
    required = ["training_m6", "treatment", "site"]
    missing = sorted(set(required) - set(data.columns))
    if missing:
        parser.error(f"Missing required columns: {', '.join(missing)}")
    if data[required].isna().any().any():
        parser.error("The analysis columns must have no missing values.")
    if set(data["treatment"].unique()) != {0, 1}:
        parser.error("Treatment must contain both 0 and 1, and no other values.")
    if not data["training_m6"].isin([0, 1]).all():
        parser.error("training_m6 must contain only 0 and 1.")

    result = smf.ols(
        "training_m6 ~ treatment + C(site)", data=data, missing="raise"
    ).fit(cov_type="HC2", use_t=True)
    if result.df_resid <= 0 or result.model.rank != result.model.exog.shape[1]:
        parser.error("The regression needs a full-rank design and residual degrees of freedom.")

    output = {
        "estimate": float(result.params["treatment"]),
        "std_error": float(result.bse["treatment"]),
        "p_value": float(result.pvalues["treatment"]),
    }
    if not all(math.isfinite(value) for value in output.values()):
        parser.error("The regression did not produce finite inference statistics.")
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
