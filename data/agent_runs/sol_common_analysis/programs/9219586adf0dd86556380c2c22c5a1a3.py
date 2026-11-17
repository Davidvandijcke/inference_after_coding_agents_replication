#!/usr/bin/env python3
"""Estimate the 12-month employment intention-to-treat effect.

Usage: python score.py --data data.csv --output result.json
Requires: numpy, pandas, scipy, statsmodels.

The sole model is OLS: employed_m12 ~ treatment + site indicators +
employed_baseline, including an intercept. Inference uses HC2 standard errors
and a two-sided Student t-test with n - rank(X) degrees of freedom.
The estimate and standard error are in probability units, not percentage points.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error("--output must differ from --data")

    columns = ["employed_m12", "treatment", "site", "employed_baseline"]
    data = pd.read_csv(args.data, usecols=columns)
    if data.empty or data.isna().any().any():
        parser.error("The analysis columns must contain complete observations")
    for column in ["employed_m12", "treatment", "employed_baseline"]:
        if not data[column].isin([0, 1]).all():
            parser.error(f"{column} must contain only 0 and 1")

    design = pd.concat(
        [
            data[["treatment", "employed_baseline"]].astype(float),
            pd.get_dummies(data["site"], prefix="site", drop_first=True, dtype=float),
        ],
        axis=1,
    )
    design = sm.add_constant(design, has_constant="add")
    if len(data) <= design.shape[1] or np.linalg.matrix_rank(design) < design.shape[1]:
        parser.error("The specified regression needs a full-rank design and residual degrees of freedom")

    fit = sm.OLS(data["employed_m12"].astype(float), design).fit(
        cov_type="HC2", use_t=True
    )
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
