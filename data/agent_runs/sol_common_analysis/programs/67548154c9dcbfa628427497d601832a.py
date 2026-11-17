#!/usr/bin/env python3
"""Estimate the assignment effect on six-month debt collections.

Usage: python score.py --data data.csv --output result.json
Dependencies: pandas, statsmodels.
The fixed analysis is OLS with an intercept, treatment, and site indicators;
HC2 standard errors and a two-sided t test use residual degrees of freedom.
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
        args.data, usecols=["treatment", "site", "debt_collections_m6"]
    )
    if data.isna().any().any():
        raise ValueError("The analysis columns must have no missing values.")
    if not data["treatment"].isin([0, 1]).all():
        raise ValueError("treatment must contain only 0 and 1.")
    if not data["debt_collections_m6"].isin([0, 1]).all():
        raise ValueError("debt_collections_m6 must contain only 0 and 1.")

    sites = pd.get_dummies(data["site"], prefix="site", drop_first=True, dtype=float)
    design = pd.concat([data[["treatment"]], sites], axis=1)
    design.insert(0, "intercept", 1.0)
    design = design.astype(float)
    # With site indicators, treatment is identified by within-site variation.
    if len(data) <= design.shape[1] or not (
        data.groupby("site")["treatment"].nunique() == 2
    ).any():
        raise ValueError("Insufficient observations or within-site treatment variation.")
    if (data.groupby("site").size() < 2).any():
        raise ValueError("HC2 requires more than one participant per site.")

    fit = sm.OLS(data["debt_collections_m6"], design).fit(
        cov_type="HC2", use_t=True
    )
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    args.output.write_text(json.dumps(result, allow_nan=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
