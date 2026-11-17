#!/usr/bin/env python3
"""Estimate the assignment effect on 12-month self-rated health.

Usage: python score.py --data data.csv --output result.json
Requires pandas and statsmodels (and their dependencies).

The sole analysis is OLS: health_m12 ~ treatment + site indicators.
Inference uses HC2 standard errors and a two-sided t test with residual
degrees of freedom. The reported p_value is the unadjusted p-value for
this fixed test; no outcome search or multiplicity adjustment is run.
"""

import argparse
import json
import math
from pathlib import Path

import pandas as pd
import statsmodels.api as sm


def score(data_path):
    data = pd.read_csv(data_path, usecols=["health_m12", "treatment", "site"])
    if data.empty or data.isna().any().any():
        raise ValueError("Analysis columns must contain complete observations.")
    if set(data["treatment"].unique()) != {0, 1}:
        raise ValueError("treatment must contain both 0 and 1.")

    sites = pd.get_dummies(data["site"], prefix="site", drop_first=True, dtype=float)
    design = sm.add_constant(
        pd.concat([data[["treatment"]], sites], axis=1), has_constant="add"
    )
    fit = sm.OLS(data["health_m12"], design).fit(cov_type="HC2", use_t=True)
    if fit.model.rank != design.shape[1] or fit.df_resid <= 0:
        raise ValueError("The treatment effect requires a full-rank design and residual degrees of freedom.")

    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("The analysis did not produce finite inference statistics.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error("--output must differ from --data")
    result = score(args.data)
    args.output.write_text(json.dumps(result, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
