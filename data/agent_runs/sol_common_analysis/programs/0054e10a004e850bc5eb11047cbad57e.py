#!/usr/bin/env python3
"""Estimate the site-adjusted ITT effect on employment at 12 months.

Requires pandas and statsmodels. The estimate and HC2 standard error are in
probability units. p_value is the unadjusted, two-sided Student t-test p-value
using the OLS residual degrees of freedom. No outcome search is performed.
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

    data = pd.read_csv(
        args.data, usecols=["site", "treatment", "employed_m12"]
    )
    if data.isna().any().any():
        raise ValueError("The analysis columns must have no missing values.")
    if set(data["treatment"].unique()) != {0, 1}:
        raise ValueError("treatment must contain both 0 and 1, and no other values.")
    if not data["employed_m12"].isin([0, 1]).all():
        raise ValueError("employed_m12 must be binary (0 or 1).")

    sites = pd.get_dummies(data["site"], prefix="site", drop_first=True, dtype=float)
    design = sm.add_constant(
        pd.concat([data[["treatment"]], sites], axis=1), has_constant="add"
    ).astype(float)
    result = sm.OLS(data["employed_m12"], design).fit(
        cov_type="HC2", use_t=True
    )
    output = {
        "estimate": float(result.params["treatment"]),
        "std_error": float(result.bse["treatment"]),
        "p_value": float(result.pvalues["treatment"]),
    }
    args.output.write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
