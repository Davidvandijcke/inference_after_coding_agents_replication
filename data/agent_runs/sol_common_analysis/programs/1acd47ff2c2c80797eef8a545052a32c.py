#!/usr/bin/env python3
"""Estimate the fixed 12-month employment ITT effect.

Requires pandas and statsmodels. The only fitted model is
employed_m12 ~ treatment + C(site), with HC2 standard errors and a
two-sided residual-df t test. The reported p_value is unadjusted.
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
        parser.error("--output must differ from --data")

    data = pd.read_csv(args.data, usecols=["employed_m12", "treatment", "site"])
    if data.empty or data.isna().any().any():
        parser.error("Analysis columns must be nonempty and fully observed")
    for column in ("employed_m12", "treatment"):
        if not data[column].isin([0, 1]).all():
            parser.error(f"{column} must contain only 0 and 1")
    if data["treatment"].nunique() != 2:
        parser.error("Both treatment groups must be present")

    model = smf.ols("employed_m12 ~ treatment + C(site)", data=data,
                    missing="raise")
    result = model.fit(cov_type="HC2", use_t=True)
    if result.model.rank != model.exog.shape[1] or result.df_resid <= 0:
        parser.error("The model must be identified with positive residual degrees of freedom")
    output = {
        "estimate": float(result.params["treatment"]),
        "std_error": float(result.bse["treatment"]),
        "p_value": float(result.pvalues["treatment"]),
    }
    if not all(math.isfinite(value) for value in output.values()):
        parser.error("The data do not support finite HC2 inference")
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
