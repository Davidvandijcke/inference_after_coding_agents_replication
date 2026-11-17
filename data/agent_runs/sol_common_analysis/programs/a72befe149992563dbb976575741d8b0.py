#!/usr/bin/env python3
"""Estimate the full-sample effect of assignment on twelve-month earnings.

Usage: python score.py --data data.csv --output results.json
Requires pandas, numpy, scipy, and statsmodels.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    data = pd.read_csv(args.data, usecols=["earnings_m12", "treatment", "site"])
    if data.empty or data.isna().any().any():
        raise ValueError("The analysis columns must contain complete observations.")
    if set(data["treatment"].unique()) != {0, 1}:
        raise ValueError("Treatment must contain both 0 and 1, with no other values.")
    outcome = pd.to_numeric(data["earnings_m12"], errors="raise").astype(float)
    sites = pd.get_dummies(data["site"], prefix="site", drop_first=True, dtype=float)
    design = sm.add_constant(
        pd.concat([data[["treatment"]].astype(float), sites], axis=1),
        has_constant="add",
    )
    if not np.isfinite(outcome).all() or not np.isfinite(design.to_numpy()).all():
        raise ValueError("The outcome and regression design must be finite.")
    if len(data) <= design.shape[1] or np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("The model needs a full-rank design and residual degrees of freedom.")

    # Only the chosen analysis: OLS with site indicators, HC2 standard errors,
    # and a two-sided Student t test using n minus the design rank degrees of freedom.
    result = sm.OLS(outcome, design).fit(cov_type="HC2", use_t=True)
    output = {
        "estimate": float(result.params["treatment"]),
        "std_error": float(result.bse["treatment"]),
        "p_value": float(result.pvalues["treatment"]),
    }
    serialized = json.dumps(output, indent=2, allow_nan=False) + "\n"
    if args.output.resolve() == args.data.resolve():
        raise ValueError("The output path must differ from the input data path.")
    args.output.write_text(serialized, encoding="utf-8")


if __name__ == "__main__":
    main()
