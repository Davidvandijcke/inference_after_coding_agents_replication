#!/usr/bin/env python3
"""Estimate the ITT effect on 12-month employment, adjusting for site.

Dependencies: pandas and statsmodels.
Inference: OLS with HC2 standard errors and a two-sided Student t test
using the regression residual degrees of freedom. No outcome or model search.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm


def analyze(data_path):
    data = pd.read_csv(data_path, usecols=["site", "treatment", "employed_m12"])
    if data.empty or data.isna().any().any():
        raise ValueError("The analysis requires nonempty, complete observations.")
    if set(data["treatment"].unique()) != {0, 1}:
        raise ValueError("Treatment must contain both 0 and 1, and no other values.")
    if not data["employed_m12"].isin([0, 1]).all():
        raise ValueError("employed_m12 must be binary (0 or 1).")

    site_indicators = pd.get_dummies(
        data["site"], prefix="site", drop_first=True, dtype=float
    )
    design = sm.add_constant(
        pd.concat([data[["treatment"]].astype(float), site_indicators], axis=1),
        has_constant="add",
    )
    result = sm.OLS(data["employed_m12"].astype(float), design).fit(
        cov_type="HC2", use_t=True
    )
    return {
        "estimate": float(result.params["treatment"]),
        "std_error": float(result.bse["treatment"]),
        "p_value": float(result.pvalues["treatment"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error("The output path must differ from the input data path.")
    output = analyze(args.data)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
