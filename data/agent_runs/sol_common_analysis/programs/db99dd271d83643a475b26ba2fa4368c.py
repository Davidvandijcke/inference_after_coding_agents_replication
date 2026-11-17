#!/usr/bin/env python3
"""Estimate the assignment effect on weekly hours at 12 months.

Requires pandas and statsmodels. Fits only the chosen site-adjusted OLS
analysis, with HC2 standard errors and a two-sided residual-df t-test.
The returned p-value is nominal (not adjusted for the original outcome search).
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

    data = pd.read_csv(args.data, usecols=["hours_m12", "treatment", "site"])
    model = smf.ols(
        "hours_m12 ~ treatment + C(site)", data=data, missing="raise"
    ).fit(cov_type="HC2", use_t=True)
    result = {
        "estimate": float(model.params["treatment"]),
        "std_error": float(model.bse["treatment"]),
        "p_value": float(model.pvalues["treatment"]),
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
