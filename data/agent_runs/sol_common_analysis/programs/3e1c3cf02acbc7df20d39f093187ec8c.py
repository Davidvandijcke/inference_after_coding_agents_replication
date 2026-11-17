#!/usr/bin/env python3
"""Estimate the 12-month employment ITT effect, adjusting for site.

Usage: python score.py --data data.csv --output result.json
Requires pandas and statsmodels. Uses OLS, HC2 standard errors, and a
two-sided t-test with residual degrees of freedom, as in CODEBOOK.md.
The estimate and standard error are in probability units (1.0 = 100 pp).
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

    data = pd.read_csv(
        args.data, usecols=["site", "treatment", "employed_m12"]
    )
    model = smf.ols(
        "employed_m12 ~ treatment + C(site)", data=data, missing="raise"
    ).fit(cov_type="HC2", use_t=True)
    result = {
        "estimate": float(model.params["treatment"]),
        "std_error": float(model.bse["treatment"]),
        "p_value": float(model.pvalues["treatment"]),
    }
    args.output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
