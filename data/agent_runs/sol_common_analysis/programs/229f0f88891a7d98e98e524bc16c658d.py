#!/usr/bin/env python3
"""Estimate the assignment effect on debt collections at 12 months.

Usage: python score.py --data data.csv --output result.json
Requires pandas, statsmodels, and their dependencies.
The p-value is the nominal two-sided HC2 t-test p-value.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = ["site", "treatment", "debt_collections_m12"]
    if data[required].isna().any().any():
        raise ValueError("The chosen analysis requires complete observations.")
    if not data["treatment"].isin([0, 1]).all():
        raise ValueError("Treatment must be coded 0 or 1.")
    sites = pd.get_dummies(data["site"], prefix="site", drop_first=True,
                           dtype=float)
    design = sm.add_constant(pd.concat([data[["treatment"]], sites], axis=1),
                             has_constant="add")
    if len(data) <= design.shape[1]:
        raise ValueError("Insufficient observations for residual degrees of freedom.")
    # Assignment must vary within at least one site to identify its effect.
    if not data.groupby("site")["treatment"].nunique().gt(1).any():
        raise ValueError("Treatment is not identifiable separately from site.")
    fit = sm.OLS(data["debt_collections_m12"], design,
                 missing="raise").fit(cov_type="HC2", use_t=True)
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
