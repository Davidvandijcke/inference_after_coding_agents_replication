#!/usr/bin/env python3
"""Estimate the full-sample effect on employment at 12 months.

Requires pandas and statsmodels. The sole model is
employed_m12 ~ treatment + C(site), with HC2 standard errors and a
two-sided Student t-test using residual degrees of freedom.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error("--output must differ from --data")

    data = pd.read_csv(args.data, usecols=["employed_m12", "treatment", "site"])
    if data.empty or data.isna().any().any():
        parser.error("Analysis columns must contain complete observations")
    if set(data["treatment"].unique()) != {0, 1}:
        parser.error("treatment must contain both 0 and 1")
    if not data["employed_m12"].isin([0, 1]).all():
        parser.error("employed_m12 must be coded 0 or 1")
    if data.groupby("site")["treatment"].nunique().max() < 2:
        parser.error("Treatment must vary within at least one site")

    result = smf.ols(
        "employed_m12 ~ treatment + C(site)", data=data, missing="raise"
    ).fit(cov_type="HC2", use_t=True)
    output = {
        "estimate": float(result.params["treatment"]),
        "std_error": float(result.bse["treatment"]),
        "p_value": float(result.pvalues["treatment"]),
    }
    args.output.write_text(json.dumps(output, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
