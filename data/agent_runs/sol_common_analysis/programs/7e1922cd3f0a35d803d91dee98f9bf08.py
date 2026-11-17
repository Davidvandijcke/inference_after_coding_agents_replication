#!/usr/bin/env python3
"""Estimate the intention-to-treat effect on 12-month employment.

Usage: python score.py --data data.csv --output result.json
Requires pandas and statsmodels. Fits only employed_m12 ~ treatment + C(site),
with HC2 standard errors and a two-sided Student t test (residual degrees of
freedom). The estimate is in probability units; p_value is unadjusted.
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

    data = pd.read_csv(args.data, usecols=["site", "treatment", "employed_m12"])
    if data.isna().any().any():
        raise ValueError("The chosen analysis requires complete observations.")
    for column in ("treatment", "employed_m12"):
        if not data[column].isin([0, 1]).all():
            raise ValueError(f"{column} must contain only 0 and 1.")
    if data["treatment"].nunique() != 2:
        raise ValueError("Both treatment and control observations are required.")

    model = smf.ols("employed_m12 ~ treatment + C(site)", data=data)
    if model.df_resid <= 0 or model.df_model + 1 != model.exog.shape[1]:
        raise ValueError("The regression requires a full-rank design and residual degrees of freedom.")
    result = model.fit(cov_type="HC2", use_t=True)
    output = {
        "estimate": float(result.params["treatment"]),
        "std_error": float(result.bse["treatment"]),
        "p_value": float(result.pvalues["treatment"]),
    }
    args.output.write_text(json.dumps(output, allow_nan=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
