#!/usr/bin/env python3
"""Estimate the assignment effect on 12-month employment.

Usage: python score.py --data data.csv --output result.json
Requires pandas and statsmodels. Fits only employed_m12 ~ treatment + C(site),
with HC2 standard errors and a two-sided Student t-test using residual degrees
of freedom. The estimate and standard error are in employment-probability units.
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
        parser.error("Output must differ from the input data path.")

    data = pd.read_csv(args.data, usecols=["employed_m12", "treatment", "site"])
    if data.isna().any().any():
        raise ValueError("Outcome, treatment, and site must all be observed.")
    if not data["employed_m12"].isin([0, 1]).all():
        raise ValueError("employed_m12 must contain only 0 and 1.")
    if set(data["treatment"].unique()) != {0, 1}:
        raise ValueError("treatment must contain both 0 and 1, and no other values.")

    fit = smf.ols("employed_m12 ~ treatment + C(site)", data=data).fit(
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
