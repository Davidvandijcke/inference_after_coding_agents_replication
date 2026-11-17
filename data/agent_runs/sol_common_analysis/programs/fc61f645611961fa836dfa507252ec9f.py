#!/usr/bin/env python3
"""Estimate the full-sample ITT effect on employment at 12 months.

Dependencies: pandas, statsmodels.
Model: employed_m12 ~ intercept + treatment + categorical site indicators.
Inference: HC2 standard error and a two-sided t-test using residual degrees
of freedom. The output p-value is nominal, without multiplicity adjustment.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.api as sm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    data = pd.read_csv(args.data, usecols=["site", "treatment", "employed_m12"])
    if data.empty or data.isna().any().any():
        raise ValueError("The analysis requires nonempty, complete observations.")
    for column in ("treatment", "employed_m12"):
        if not data[column].isin([0, 1]).all():
            raise ValueError(f"{column} must contain only 0 and 1.")
    if data["treatment"].nunique() != 2:
        raise ValueError("Both treatment arms must be present.")
    if (data.groupby("site")["treatment"].nunique() != 2).any():
        raise ValueError("Both treatment arms must be present within each site.")

    site_indicators = pd.get_dummies(
        data["site"], prefix="site", drop_first=True, dtype=float
    )
    design = sm.add_constant(
        pd.concat([data[["treatment"]].astype(float), site_indicators], axis=1),
        has_constant="add",
    )
    if len(data) <= design.shape[1]:
        raise ValueError("Insufficient residual degrees of freedom.")
    result = sm.OLS(data["employed_m12"], design, missing="raise").fit(
        cov_type="HC2", use_t=True
    )
    output = {
        "estimate": float(result.params["treatment"]),
        "std_error": float(result.bse["treatment"]),
        "p_value": float(result.pvalues["treatment"]),
    }
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
