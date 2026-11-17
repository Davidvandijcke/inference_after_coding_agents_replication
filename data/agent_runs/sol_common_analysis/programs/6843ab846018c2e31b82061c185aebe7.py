#!/usr/bin/env python3
"""Estimate the 12-month employment intention-to-treat effect.

Requires pandas and statsmodels. Fits OLS with treatment and site indicators,
HC2 standard errors, and a two-sided t-test using residual degrees of freedom.
Usage: python score.py --data data.csv --output result.json
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
    if data.empty or data.isna().any().any():
        parser.error("The selected analysis requires complete, nonempty data")
    if set(data["treatment"].unique()) != {0, 1}:
        parser.error("treatment must contain both 0 and 1")
    if not data["employed_m12"].isin([0, 1]).all():
        parser.error("employed_m12 must contain only 0 and 1")

    sites = pd.get_dummies(data["site"], prefix="site", drop_first=True,
                           dtype=float)
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
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
