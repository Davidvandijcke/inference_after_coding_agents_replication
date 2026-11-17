#!/usr/bin/env python3
"""Estimate the assignment effect on employment at 12 months.

Model: employed_m12 ~ treatment + intercept + site indicators.
Inference: HC2 standard error and a two-sided Student t test with n - rank(X)
degrees of freedom. Estimates are in probability units, not percentage points.
Requires pandas and statsmodels. No other outcomes or specifications are run.
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
        parser.error("--output must be different from --data")

    data = pd.read_csv(args.data, usecols=["employed_m12", "treatment", "site"])
    if data.empty or data.isna().any().any():
        parser.error("The analysis requires nonempty, complete observations")
    if not data["employed_m12"].isin([0, 1]).all():
        parser.error("employed_m12 must be binary (0 or 1)")
    if set(data["treatment"].unique()) != {0, 1}:
        parser.error("treatment must contain both 0 and 1, and no other values")
    if not data.groupby("site")["treatment"].nunique().eq(2).all():
        parser.error("Both treatment arms must be represented within every site")

    sites = pd.get_dummies(data["site"], prefix="site", drop_first=True, dtype=float)
    design = pd.concat([data[["treatment"]].astype(float), sites], axis=1)
    design = sm.add_constant(design, has_constant="add")
    if len(data) <= design.shape[1]:
        parser.error("Insufficient residual degrees of freedom")
    fit = sm.OLS(data["employed_m12"].astype(float), design).fit(
        cov_type="HC2", use_t=True
    )
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
