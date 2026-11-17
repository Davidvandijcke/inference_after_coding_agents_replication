#!/usr/bin/env python3
"""Estimate the six-month earnings ITT using site-adjusted OLS and HC2.

Usage: python score.py --data data.csv --output result.json
Requires pandas and statsmodels. The p-value is the nominal two-sided
Student t-test of the treatment coefficient, with residual degrees of freedom.
Only earnings_m6 is analyzed; no outcome or subgroup selection is performed.
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

    data = pd.read_csv(args.data, usecols=["site", "treatment", "earnings_m6"])
    if data.isna().any().any():
        parser.error("site, treatment, and earnings_m6 must have no missing values")
    if set(data["treatment"].unique()) != {0, 1}:
        parser.error("treatment must contain both 0 and 1, and no other values")

    sites = pd.get_dummies(data["site"], prefix="site", drop_first=True, dtype=float)
    design = sm.add_constant(
        pd.concat([data[["treatment"]], sites], axis=1), has_constant="add"
    ).astype(float)
    result = sm.OLS(data["earnings_m6"].astype(float), design).fit(
        cov_type="HC2", use_t=True
    )
    output = {
        "estimate": float(result.params["treatment"]),
        "std_error": float(result.bse["treatment"]),
        "p_value": float(result.pvalues["treatment"]),
    }
    serialized = json.dumps(output, indent=2, allow_nan=False) + "\n"
    args.output.write_text(serialized, encoding="utf-8")


if __name__ == "__main__":
    main()
