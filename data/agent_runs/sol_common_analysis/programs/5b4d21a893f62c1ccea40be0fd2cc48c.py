#!/usr/bin/env python3
"""Estimate the 12-month debt-collections intention-to-treat effect.

Usage: python score.py --data data.csv --output result.json
Requires pandas and statsmodels. Fits only debt_collections_m12 on an intercept,
treatment, and categorical site indicators. Uses HC2 standard errors and a
two-sided Student t test with residual degrees of freedom. The p_value is the
nominal p-value for this single, fixed analysis, without selection adjustment.
"""

import argparse
import json

import pandas as pd
import statsmodels.formula.api as smf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data = pd.read_csv(
        args.data, usecols=["debt_collections_m12", "treatment", "site"]
    )
    if data.isna().any().any():
        raise ValueError("The analysis columns must have no missing values.")
    model = smf.ols(
        "debt_collections_m12 ~ treatment + C(site)", data=data,
        missing="raise",
    ).fit(cov_type="HC2", use_t=True)
    result = {
        "estimate": float(model.params["treatment"]),
        "std_error": float(model.bse["treatment"]),
        "p_value": float(model.pvalues["treatment"]),
    }
    with open(args.output, "w", encoding="utf-8") as output:
        json.dump(result, output, allow_nan=False, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
