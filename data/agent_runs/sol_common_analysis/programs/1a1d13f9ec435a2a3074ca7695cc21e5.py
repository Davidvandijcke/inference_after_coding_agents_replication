#!/usr/bin/env python3
"""Estimate the intent-to-treat effect on employment at 12 months.

Usage: python score.py --data data.csv --output result.json
Requires pandas and statsmodels. Fits only the selected model: OLS of
employed_m12 on an intercept, treatment, and categorical site indicators,
with HC2 standard errors and a two-sided Student t test (residual df).
Effects and standard errors are in employment-probability units.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.data.resolve() == args.output.resolve():
        parser.error("--output must differ from --data")

    data = pd.read_csv(args.data, usecols=["treatment", "site", "employed_m12"])
    if data.isna().any().any():
        parser.error("treatment, site, and employed_m12 must be fully observed")
    if set(data["treatment"].unique()) != {0, 1}:
        parser.error("treatment must contain both 0 and 1, and no other values")
    if not data["employed_m12"].isin([0, 1]).all():
        parser.error("employed_m12 must contain only 0 and 1")
    if not data.groupby("site")["treatment"].nunique().eq(2).any():
        parser.error("at least one site must contain both treatment arms")

    site_indicators = pd.get_dummies(
        data["site"], prefix="site", drop_first=True, dtype=float
    )
    design = sm.add_constant(
        pd.concat([data[["treatment"]], site_indicators], axis=1),
        has_constant="add",
    ).astype(float)
    if len(data) <= design.shape[1]:
        parser.error("insufficient observations for residual degrees of freedom")

    fit = sm.OLS(data["employed_m12"], design).fit(cov_type="HC2", use_t=True)
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
