#!/usr/bin/env python3
"""Estimate the 12-month employment ITT using site fixed effects and HC2.

Usage: python score.py --data data.csv --output results.json
Requires numpy, pandas, scipy, and statsmodels. Estimates and standard errors
are in probability units; p_value is a nominal two-sided Student t-test.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def analyze(data_path):
    data = pd.read_csv(data_path, usecols=["site", "treatment", "employed_m12"])
    if data.empty or data.isna().any().any():
        raise ValueError("The analysis requires nonempty, complete observations.")
    if not data["treatment"].isin([0, 1]).all():
        raise ValueError("treatment must contain only 0 and 1.")
    if data["treatment"].nunique() != 2:
        raise ValueError("Both treatment groups must be present.")
    if not data["employed_m12"].isin([0, 1]).all():
        raise ValueError("employed_m12 must contain only 0 and 1.")

    site_indicators = pd.get_dummies(
        data["site"], prefix="site", drop_first=True, dtype=float
    )
    design = sm.add_constant(
        pd.concat([data[["treatment"]].astype(float), site_indicators], axis=1),
        has_constant="add",
    )
    if len(data) <= design.shape[1]:
        raise ValueError("Insufficient observations for residual degrees of freedom.")
    if np.linalg.matrix_rank(design.to_numpy()) != design.shape[1]:
        raise ValueError("Treatment and site effects are not separately identifiable.")
    # HC2 is the codebook's robust variance; use_t=True explicitly selects
    # Student t inference with n - rank(design) residual degrees of freedom.
    fit = sm.OLS(data["employed_m12"].astype(float), design).fit(
        cov_type="HC2", use_t=True
    )
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("The requested inference is undefined for this dataset.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = analyze(args.data)
    # Protect the input observations from an accidental output-path collision.
    if args.data.resolve() == args.output.resolve():
        parser.error("--output must differ from --data")
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
