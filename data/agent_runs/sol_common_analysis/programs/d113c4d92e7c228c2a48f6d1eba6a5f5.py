#!/usr/bin/env python3
"""Estimate the 12-month earnings ITT, adjusting for randomization site.

Usage: python score.py --data data.csv --output result.json
Dependencies: numpy, pandas, scipy, statsmodels.
Model: earnings_m12 ~ intercept + treatment + categorical site indicators.
Inference: HC2 standard error and a two-sided Student t test, with n - rank(X)
degrees of freedom. No subgroup selection or other outcome analyses are run.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def score(data_path):
    data = pd.read_csv(data_path, usecols=["treatment", "site", "earnings_m12"])
    if data.empty or data.isna().any().any():
        raise ValueError("The analysis requires nonempty, complete observations.")
    if not data["treatment"].isin([0, 1]).all():
        raise ValueError("treatment must be coded 0 or 1.")

    outcome = data["earnings_m12"].astype(float)
    design = pd.concat(
        [
            data[["treatment"]].astype(float),
            pd.get_dummies(data["site"], prefix="site", drop_first=True, dtype=float),
        ],
        axis=1,
    )
    design = sm.add_constant(design, has_constant="add")
    if not np.isfinite(outcome.to_numpy()).all():
        raise ValueError("earnings_m12 must contain only finite numbers.")
    if len(data) <= design.shape[1] or np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("The site-adjusted treatment effect is not estimable.")

    fit = sm.OLS(outcome, design).fit(cov_type="HC2", use_t=True)
    return {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error("--output must differ from --data")
    result = score(args.data)
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
