#!/usr/bin/env python3
"""Estimate the 12-month employment ITT using site-adjusted OLS and HC2.

Usage: python score.py --data data.csv --output result.json
Dependencies: numpy, pandas, statsmodels.
The p_value is the unadjusted, two-sided Student t-test with residual df.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def score(data_path):
    data = pd.read_csv(data_path, usecols=["treatment", "site", "employed_m12"])
    if data.empty or data.isna().any().any():
        raise ValueError("Analysis columns must be nonempty and fully observed.")
    if set(data["treatment"].unique()) != {0, 1}:
        raise ValueError("Treatment must contain both 0 and 1, and no other values.")
    if not data["employed_m12"].isin([0, 1]).all():
        raise ValueError("employed_m12 must be binary (0 or 1).")

    model = smf.ols("employed_m12 ~ treatment + C(site)", data=data)
    design = model.exog
    if len(data) <= design.shape[1] or np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("Insufficient observations or an unidentified treatment effect.")
    result = model.fit(cov_type="HC2", use_t=True)
    output = {
        "estimate": float(result.params["treatment"]),
        "std_error": float(result.bse["treatment"]),
        "p_value": float(result.pvalues["treatment"]),
    }
    if not all(np.isfinite(value) for value in output.values()):
        raise ValueError("The model did not produce finite estimates and inference.")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error("The output path must differ from the input data path.")
    result = score(args.data)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
