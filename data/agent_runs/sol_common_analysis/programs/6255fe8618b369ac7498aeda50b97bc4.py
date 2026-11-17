#!/usr/bin/env python3
"""Estimate the assignment effect on monthly earnings at 12 months.

Usage: python score.py --data data.csv --output result.json
Dependencies: numpy, pandas, scipy, statsmodels.

OLS includes an intercept, treatment, site indicators (one omitted), baseline
monthly earnings, and baseline employment. Inference uses HC2 standard errors
and a two-sided Student t test with the regression residual degrees of freedom.
No outcome selection, subgroup search, or other analyses are performed.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def score(data_path):
    columns = [
        "site",
        "treatment",
        "earnings_baseline",
        "employed_baseline",
        "earnings_m12",
    ]
    data = pd.read_csv(data_path, usecols=columns)
    if data.empty or data.isna().any().any():
        raise ValueError("The analysis requires nonempty, complete data.")
    numeric = data[[c for c in columns if c != "site"]].astype(float)
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Analysis variables must contain finite numbers.")
    if set(numeric["treatment"].unique()) != {0.0, 1.0}:
        raise ValueError("Both treatment assignments, coded 0 and 1, are required.")

    design = pd.concat(
        [
            numeric[["treatment", "earnings_baseline", "employed_baseline"]],
            pd.get_dummies(data["site"], prefix="site", drop_first=True, dtype=float),
        ],
        axis=1,
    )
    design = sm.add_constant(design, has_constant="add").astype(float)
    fit = sm.OLS(numeric["earnings_m12"], design, missing="raise").fit(
        cov_type="HC2", use_t=True
    )
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
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
