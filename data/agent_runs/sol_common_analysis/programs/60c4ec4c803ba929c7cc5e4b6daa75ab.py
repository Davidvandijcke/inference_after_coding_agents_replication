#!/usr/bin/env python3
"""Estimate the intention-to-treat effect on monthly earnings at 12 months.

Usage: python score.py --data data.csv --output result.json
Dependencies: pandas, statsmodels.
The reported p-value is the unadjusted, two-sided HC2 t-test p-value.
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

    data = pd.read_csv(args.data)
    required = ["earnings_m12", "treatment", "site"]
    if data[required].isna().any().any():
        raise ValueError("The chosen analysis requires complete outcome, treatment, and site data.")
    if set(data["treatment"].unique()) != {0, 1}:
        raise ValueError("Treatment must contain both 0 and 1.")

    fit = smf.ols(
        "earnings_m12 ~ treatment + C(site)", data=data, missing="raise"
    ).fit(cov_type="HC2", use_t=True)
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
