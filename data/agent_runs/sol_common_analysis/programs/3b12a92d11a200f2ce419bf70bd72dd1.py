#!/usr/bin/env python3
"""Estimate the intention-to-treat effect on employment at 12 months.

Usage: python score.py --data data.csv --output result.json
Requires pandas and statsmodels. Uses OLS with site fixed effects, HC2
standard errors, and a two-sided t-test with residual degrees of freedom.
No outcome screening, subgroup selection, or additional analyses are run.
"""

import argparse
import json
import math
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error("The output path must differ from the input data path.")

    columns = ["employed_m12", "treatment", "site"]
    data = pd.read_csv(args.data, usecols=columns)
    if data.empty or data.isna().any().any():
        raise ValueError("The analysis columns must be nonempty and fully observed.")
    if set(data["treatment"].unique()) != {0, 1}:
        raise ValueError("Treatment must contain both 0 and 1 and no other values.")
    if not data["employed_m12"].isin([0, 1]).all():
        raise ValueError("Employment at 12 months must be coded 0 or 1.")

    model = smf.ols("employed_m12 ~ treatment + C(site)", data=data,
                    missing="raise")
    fit = model.fit(cov_type="HC2", use_t=True)
    if fit.df_resid <= 0 or fit.model.rank != len(fit.params):
        raise ValueError("The regression must be identified with positive residual degrees of freedom.")
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("The dataset does not yield finite estimates and inference.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
