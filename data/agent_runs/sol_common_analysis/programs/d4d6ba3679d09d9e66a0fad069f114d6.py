#!/usr/bin/env python3
"""Estimate the intention-to-treat effect on 12-month employment.

Usage: python score.py --data data.csv --output result.json
Dependencies: numpy, pandas, statsmodels.
Only fits employed_m12 ~ treatment + C(site), with HC2 standard errors
and a two-sided t-test using residual degrees of freedom. The reported
p_value is the unadjusted p-value for this fixed test.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error("The output path must differ from the input data path.")

    data = pd.read_csv(args.data, usecols=["site", "treatment", "employed_m12"])
    if data.empty or data.isna().any().any():
        parser.error("The three analysis columns must contain complete observations.")
    for column in ("treatment", "employed_m12"):
        data[column] = pd.to_numeric(data[column], errors="raise")
        if not data[column].isin([0, 1]).all():
            parser.error(f"{column} must contain only 0 and 1.")
    if data["treatment"].nunique() != 2:
        parser.error("Both treatment groups are required.")

    model = smf.ols("employed_m12 ~ treatment + C(site)", data=data)
    design = model.exog
    if len(data) <= design.shape[1] or np.linalg.matrix_rank(design) < design.shape[1]:
        parser.error("The regression requires a full-rank design and positive residual degrees of freedom.")
    leverage = np.sum(np.linalg.qr(design, mode="reduced")[0] ** 2, axis=1)
    if np.any(leverage >= 1 - 1e-12):
        parser.error("HC2 standard errors require observation leverage below one.")

    result = model.fit(cov_type="HC2", use_t=True)
    output = {
        "estimate": float(result.params["treatment"]),
        "std_error": float(result.bse["treatment"]),
        "p_value": float(result.pvalues["treatment"]),
    }
    if not all(np.isfinite(value) for value in output.values()):
        parser.error("The chosen test did not produce finite results.")
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
