#!/usr/bin/env python3
"""Estimate the site-adjusted ITT effect on hours_m6.

Requires pandas and statsmodels. The reported p-value is the nominal,
two-sided HC2 t-test for the treatment coefficient, with n - rank(X)
residual degrees of freedom. This program fits only this one model.
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

    data = pd.read_csv(args.data, usecols=["hours_m6", "treatment", "site"])
    fit = smf.ols(
        "hours_m6 ~ treatment + C(site)", data=data, missing="raise"
    ).fit(cov_type="HC2", use_t=True)
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
