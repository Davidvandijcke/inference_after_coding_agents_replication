#!/usr/bin/env python3
"""Estimate the intention-to-treat effect on employment at 12 months.

Usage: python score.py --data data.csv --output result.json
Requires pandas and statsmodels. Estimates are in probability units.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error("The output path must differ from the input data path.")

    data = pd.read_csv(args.data, usecols=["treatment", "site", "employed_m12"])
    if data.empty or data.isna().any().any():
        parser.error("Analysis columns must contain observations with no missing values.")
    if set(data["treatment"].unique()) != {0, 1}:
        parser.error("Treatment must contain both 0 and 1, and no other values.")
    if not data["employed_m12"].isin([0, 1]).all():
        parser.error("Employment at 12 months must be coded 0 or 1.")

    sites = pd.get_dummies(data["site"], prefix="site", drop_first=True, dtype=float)
    design = sm.add_constant(
        pd.concat([data[["treatment"]].astype(float), sites], axis=1),
        has_constant="add",
    )
    # One full-sample regression, no covariate or outcome selection.
    fit = sm.OLS(data["employed_m12"].astype(float), design).fit(
        cov_type="HC2", use_t=True
    )
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
    args.output.write_text(serialized, encoding="utf-8")


if __name__ == "__main__":
    main()
