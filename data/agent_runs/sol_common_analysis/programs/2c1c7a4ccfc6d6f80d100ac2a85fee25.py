#!/usr/bin/env python3
"""Estimate the effect of assignment on employment at 12 months.

Usage: python score.py --data data.csv --output result.json
Requires pandas and statsmodels (and their dependencies).

The single model is employed_m12 ~ treatment + C(site), fitted by OLS.
Inference uses HC2 standard errors and a two-sided t test with the OLS
residual degrees of freedom. No covariate selection or subgroup search runs.
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
        parser.error("The output must be different from the input data.")

    data = pd.read_csv(args.data, usecols=["site", "treatment", "employed_m12"])
    if data.isna().any().any():
        parser.error("Site, treatment, and employed_m12 must be fully observed.")
    if set(data["treatment"].unique()) != {0, 1}:
        parser.error("Treatment must contain both 0 and 1, and no other values.")
    if not data["employed_m12"].isin([0, 1]).all():
        parser.error("Employed_m12 must contain only 0 and 1.")

    fit = smf.ols("employed_m12 ~ treatment + C(site)", data=data).fit(
        cov_type="HC2", use_t=True
    )
    if fit.model.rank != fit.model.exog.shape[1] or fit.df_resid <= 0:
        parser.error("The treatment effect requires an identifiable model with residual degrees of freedom.")

    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    if not all(math.isfinite(value) for value in result.values()):
        parser.error("The model did not produce finite estimates and inference.")
    with args.output.open("w", encoding="utf-8") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
