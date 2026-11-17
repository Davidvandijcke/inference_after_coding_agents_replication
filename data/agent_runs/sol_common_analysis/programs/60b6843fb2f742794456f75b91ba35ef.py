#!/usr/bin/env python3
"""Estimate the assignment effect on 12-month employment.

Requires pandas and statsmodels. Fits only OLS of employed_m12 on an
intercept, treatment, and site indicators, with HC2 standard errors and
a two-sided Student t test using residual degrees of freedom.
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

    data = pd.read_csv(args.data, usecols=["site", "treatment", "employed_m12"])
    if data.empty or data.isna().any().any():
        raise ValueError("The selected columns must contain complete observations.")
    if set(data["treatment"].unique()) != {0, 1}:
        raise ValueError("Treatment must contain both 0 and 1.")
    if not data["employed_m12"].isin([0, 1]).all():
        raise ValueError("employed_m12 must be binary (0 or 1).")
    counts = pd.crosstab(data["site"], data["treatment"])
    if (counts == 0).any().any():
        raise ValueError("Each site must contain both treatment arms.")

    design = sm.add_constant(
        pd.concat(
            [
                data[["treatment"]].astype(float),
                pd.get_dummies(data["site"], prefix="site", drop_first=True,
                               dtype=float),
            ],
            axis=1,
        ),
        has_constant="add",
    )
    if len(data) <= design.shape[1]:
        raise ValueError("Positive residual degrees of freedom are required.")
    fit = sm.OLS(data["employed_m12"], design).fit(cov_type="HC2", use_t=True)
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
