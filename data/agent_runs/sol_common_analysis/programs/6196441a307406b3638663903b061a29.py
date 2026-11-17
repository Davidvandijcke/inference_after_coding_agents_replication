#!/usr/bin/env python3
"""Estimate the site-adjusted intention-to-treat effect on employment at 12 months.

Requires pandas and statsmodels. Reports a two-sided HC2 t-test, with residual
degrees of freedom. The p-value is nominal (not adjusted for outcome selection).
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm


def score(data_path):
    data = pd.read_csv(
        data_path, usecols=["site", "treatment", "employed_m12"]
    )
    if data.empty or data.isna().any().any():
        raise ValueError("Analysis columns must be nonempty and fully observed.")
    for column in ("treatment", "employed_m12"):
        if not data[column].isin([0, 1]).all():
            raise ValueError(f"{column} must contain only 0 and 1.")
    if data.groupby("site")["treatment"].nunique().lt(2).any():
        raise ValueError("Each site must contain both treatment assignments.")

    site_indicators = pd.get_dummies(
        data["site"], prefix="site", drop_first=True, dtype=float
    )
    design = sm.add_constant(
        pd.concat([data[["treatment"]], site_indicators], axis=1),
        has_constant="add",
    ).astype(float)
    if len(data) <= design.shape[1] or (design.sum(axis=0) <= 1).any():
        raise ValueError("Insufficient observations to estimate HC2 uncertainty.")
    fit = sm.OLS(data["employed_m12"], design, missing="raise").fit(
        cov_type="HC2", use_t=True
    )
    return {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = score(args.data)
    if args.output.resolve() == args.data.resolve():
        raise ValueError("Output must differ from the input data path.")
    args.output.write_text(json.dumps(result, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
