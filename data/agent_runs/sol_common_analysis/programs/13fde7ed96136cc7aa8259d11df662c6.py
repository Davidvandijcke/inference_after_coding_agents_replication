#!/usr/bin/env python3
"""Estimate the 12-month employment intention-to-treat effect.

Usage: python score.py --data data.csv --output result.json
Requires pandas and statsmodels. Fits only employed_m12 on an intercept,
treatment, and site indicators, with HC2 standard errors and a two-sided
Student t test using residual degrees of freedom. Values are in employment
probability units; p_value is the unadjusted p-value for this single test.
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
        parser.error("--output must differ from --data")

    data = pd.read_csv(args.data, usecols=["site", "treatment", "employed_m12"])
    if data.isna().any().any():
        raise ValueError("Site, treatment, and employed_m12 must be fully observed")
    if set(data["treatment"].unique()) != {0, 1}:
        raise ValueError("Treatment must contain both 0 and 1, with no other values")
    if not data["employed_m12"].isin([0, 1]).all():
        raise ValueError("employed_m12 must be coded 0 or 1")

    sites = pd.get_dummies(data["site"], prefix="site", drop_first=True, dtype=float)
    design = sm.add_constant(
        pd.concat([data[["treatment"]].astype(float), sites], axis=1),
        has_constant="add",
    )
    fitted = sm.OLS(data["employed_m12"], design).fit(cov_type="HC2", use_t=True)
    result = {
        "estimate": float(fitted.params["treatment"]),
        "std_error": float(fitted.bse["treatment"]),
        "p_value": float(fitted.pvalues["treatment"]),
    }
    args.output.write_text(json.dumps(result, allow_nan=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
