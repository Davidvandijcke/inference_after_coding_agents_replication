#!/usr/bin/env python3
"""Estimate the assignment effect on 12-month financial security.

Usage: python score.py --data data.csv --output result.json
Requires pandas and statsmodels. Fits only the fixed chosen analysis:
OLS(fin_security_m12 ~ treatment + site indicators), with an intercept,
HC2 standard errors, and a two-sided t-test using residual degrees of freedom.
The returned p_value is the nominal p-value for this single coefficient.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def analyze(data_path):
    data = pd.read_csv(data_path)
    required = ["fin_security_m12", "treatment", "site"]
    missing = set(required).difference(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
    if data[required].isna().any().any():
        raise ValueError("The analysis columns must be fully observed.")
    if not data["treatment"].isin([0, 1]).all():
        raise ValueError("Treatment must be coded 0 or 1.")

    sites = pd.get_dummies(data["site"], prefix="site", drop_first=True,
                           dtype=float)
    design = pd.concat([data[["treatment"]].astype(float), sites], axis=1)
    design.insert(0, "intercept", 1.0)
    outcome = data["fin_security_m12"].astype(float)
    if not np.isfinite(outcome.to_numpy()).all():
        raise ValueError("The outcome must be finite.")
    if len(data) <= design.shape[1]:
        raise ValueError("Insufficient residual degrees of freedom.")
    if np.linalg.matrix_rank(design.to_numpy()) != design.shape[1]:
        raise ValueError("Treatment and site coefficients are not identifiable.")

    fit = sm.OLS(outcome, design, missing="raise").fit(
        cov_type="HC2", use_t=True
    )
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("The specified analysis produced non-finite results.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error("The output path must differ from the input data path.")
    result = analyze(args.data)
    args.output.write_text(json.dumps(result, allow_nan=False) + "\n",
                           encoding="utf-8")


if __name__ == "__main__":
    main()
