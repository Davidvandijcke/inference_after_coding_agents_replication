#!/usr/bin/env python3
"""Estimate the effect of assignment on six-month training enrollment.

Requires pandas and statsmodels. Fits only the selected analysis: OLS of
training_m6 on an intercept, treatment, and site indicators, with HC2 standard
errors and a two-sided Student t test using the residual degrees of freedom.
The returned p_value is the nominal p-value for this one test.
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

    data = pd.read_csv(args.data, usecols=["site", "treatment", "training_m6"])
    design = pd.concat(
        [
            pd.Series(1.0, index=data.index, name="intercept"),
            data[["treatment"]].astype(float),
            pd.get_dummies(data["site"], prefix="site", drop_first=True, dtype=float),
        ],
        axis=1,
    )
    fit = sm.OLS(data["training_m6"].astype(float), design, missing="raise").fit(
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
