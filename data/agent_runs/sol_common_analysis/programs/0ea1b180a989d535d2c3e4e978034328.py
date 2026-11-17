#!/usr/bin/env python3
"""Estimate the site-adjusted ITT effect on employment at 12 months.

Requires pandas and statsmodels. Outputs the treatment coefficient (in
probability units), HC2 standard error, and nominal two-sided t-test p-value.
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

    data = pd.read_csv(args.data, usecols=["site", "treatment", "employed_m12"])
    if data.isna().any().any():
        raise ValueError("The chosen analysis requires complete observations.")
    if set(data["treatment"].unique()) != {0, 1}:
        raise ValueError("Treatment must contain both 0 and 1.")
    if not data["employed_m12"].isin([0, 1]).all():
        raise ValueError("employed_m12 must be binary (0 or 1).")

    site_indicators = pd.get_dummies(
        data["site"], prefix="site", drop_first=True, dtype=float
    )
    design = sm.add_constant(
        pd.concat([data[["treatment"]], site_indicators], axis=1),
        has_constant="add",
    )
    result = sm.OLS(data["employed_m12"], design, missing="raise").fit(
        cov_type="HC2", use_t=True
    )
    output = {
        "estimate": float(result.params["treatment"]),
        "std_error": float(result.bse["treatment"]),
        "p_value": float(result.pvalues["treatment"]),
    }
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
