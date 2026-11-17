#!/usr/bin/env python3
"""Estimate the assignment effect on 12-month employment, adjusting for site.

Usage: python score.py --data data.csv --output result.json
Requires pandas and statsmodels. Inference uses HC2 standard errors and a
two-sided Student t test with the OLS residual degrees of freedom.
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

    data = pd.read_csv(
        args.data, usecols=["site", "treatment", "employed_m12"]
    )
    if data.empty or data.isna().any().any():
        raise ValueError("The analysis requires complete, nonempty observations.")
    if set(data["treatment"].unique()) != {0, 1}:
        raise ValueError("Treatment must contain both 0 and 1.")
    if not data["employed_m12"].isin([0, 1]).all():
        raise ValueError("employed_m12 must be binary (0 or 1).")

    site_indicators = pd.get_dummies(
        data["site"], prefix="site", drop_first=True, dtype=float
    )
    design = sm.add_constant(
        pd.concat([data[["treatment"]].astype(float), site_indicators], axis=1),
        has_constant="add",
    )
    if data.groupby("site")["treatment"].nunique().max() < 2:
        raise ValueError("Treatment is not identifiable within sites.")
    if len(data) <= design.shape[1]:
        raise ValueError("There must be positive residual degrees of freedom.")

    fit = sm.OLS(data["employed_m12"], design).fit(cov_type="HC2", use_t=True)
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    args.output.write_text(json.dumps(result, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
