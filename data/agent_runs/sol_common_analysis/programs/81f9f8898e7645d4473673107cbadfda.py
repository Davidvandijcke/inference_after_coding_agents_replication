#!/usr/bin/env python3
"""Estimate the six-month employment intention-to-treat effect.

Requires pandas and statsmodels. The reported p_value is the unadjusted,
two-sided t-test p-value, using HC2 standard errors and residual degrees
of freedom, as specified in CODEBOOK.md. Effects are in probability units.
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

    data = pd.read_csv(args.data)
    required = ["employed_m6", "treatment", "site"]
    if data[required].isna().any().any():
        raise ValueError("The analysis columns must be fully observed.")
    if not data["treatment"].isin([0, 1]).all() or data["treatment"].nunique() != 2:
        raise ValueError("Treatment must contain both 0 and 1 only.")
    if not data["employed_m6"].isin([0, 1]).all():
        raise ValueError("Six-month employment must be coded 0 or 1.")

    fit = smf.ols("employed_m6 ~ treatment + C(site)", data=data).fit(
        cov_type="HC2", use_t=True
    )
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
