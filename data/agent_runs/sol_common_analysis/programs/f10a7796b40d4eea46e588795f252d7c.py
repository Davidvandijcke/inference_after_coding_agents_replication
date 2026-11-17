#!/usr/bin/env python3
"""Estimate the six-month distress ITT effect using OLS and HC2 inference.

Requires pandas and statsmodels. The two-sided p-value uses the residual-df
Student t distribution, as specified in CODEBOOK.md. It is the nominal
p-value for this fixed test, without the exploratory screen's adjustment.
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

    data = pd.read_csv(args.data, usecols=["distress_m6", "treatment", "site"])
    if data.isna().any().any():
        raise ValueError("The analysis columns must be fully observed.")
    if set(data["treatment"].unique()) != {0, 1}:
        raise ValueError("Treatment must contain both 0 and 1.")
    result = smf.ols("distress_m6 ~ treatment + C(site)", data=data).fit(
        cov_type="HC2", use_t=True
    )
    output = {
        "estimate": float(result.params["treatment"]),
        "std_error": float(result.bse["treatment"]),
        "p_value": float(result.pvalues["treatment"]),
    }
    args.output.write_text(json.dumps(output, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
