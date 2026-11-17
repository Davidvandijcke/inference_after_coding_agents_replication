#!/usr/bin/env python3
"""Estimate the 12-month employment ITT using site-adjusted OLS and HC2.

Usage: python score.py --data data.csv --output result.json
Requires pandas and statsmodels. The reported p-value is the unadjusted,
two-sided Student t-test with residual degrees of freedom.
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
    if not data["treatment"].isin([0, 1]).all() or data["treatment"].nunique() != 2:
        raise ValueError("Treatment must contain both 0 and 1.")
    if not data["employed_m12"].isin([0, 1]).all():
        raise ValueError("employed_m12 must be binary (0 or 1).")

    sites = pd.get_dummies(data["site"], prefix="site", drop_first=True, dtype=float)
    design = sm.add_constant(pd.concat([data[["treatment"]], sites], axis=1),
                             has_constant="add")
    fit = sm.OLS(data["employed_m12"], design, missing="raise").fit(
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
