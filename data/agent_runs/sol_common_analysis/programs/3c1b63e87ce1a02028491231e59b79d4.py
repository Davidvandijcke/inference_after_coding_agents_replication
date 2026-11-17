#!/usr/bin/env python3
"""Estimate the 12-month employment intention-to-treat effect.

Usage: python score.py --data data.csv --output result.json
Requires pandas and statsmodels. Uses OLS with an intercept and categorical
site indicators, HC2 standard errors, and a two-sided Student t test with
the regression's residual degrees of freedom. Estimates are in probability
units (multiply by 100 for percentage points).
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if Path(args.data).resolve() == Path(args.output).resolve():
        parser.error("--output must differ from --data")
    data = pd.read_csv(args.data)
    required = ["employed_m12", "treatment", "site"]
    if data[required].isna().any().any():
        raise ValueError("The chosen analysis requires complete outcome, treatment, and site data.")
    if set(data["treatment"].unique()) != {0, 1}:
        raise ValueError("Treatment must contain both 0 and 1.")
    if not data["employed_m12"].isin([0, 1]).all():
        raise ValueError("employed_m12 must be binary (0 or 1).")

    model = smf.ols(
        "employed_m12 ~ treatment + C(site)", data=data, missing="raise"
    ).fit(cov_type="HC2", use_t=True)
    result = {
        "estimate": float(model.params["treatment"]),
        "std_error": float(model.bse["treatment"]),
        "p_value": float(model.pvalues["treatment"]),
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
