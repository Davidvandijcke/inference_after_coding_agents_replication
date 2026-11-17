#!/usr/bin/env python3
"""Estimate the 12-month employment intention-to-treat effect.

Usage: python score.py --data data.csv --output result.json
Requires numpy, pandas, and statsmodels (and their dependencies).
The model is employed_m12 ~ treatment + C(site), with HC2 standard
errors and a two-sided t-test using residual degrees of freedom.
Effects and standard errors are in employment-probability units.
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

    data = pd.read_csv(args.data, usecols=["employed_m12", "treatment", "site"])
    if data.empty or data.isna().any().any():
        raise ValueError("Analysis columns must be nonempty and fully observed.")
    if set(data["treatment"].unique()) != {0, 1}:
        raise ValueError("Treatment must contain both 0 and 1, and no other values.")
    if not data["employed_m12"].isin([0, 1]).all():
        raise ValueError("employed_m12 must contain only 0 and 1.")

    model = smf.ols("employed_m12 ~ treatment + C(site)", data=data, missing="raise")
    rank = np.linalg.matrix_rank(model.exog)
    if rank != model.exog.shape[1] or len(data) <= rank:
        raise ValueError("The regression needs full rank and positive residual degrees of freedom.")
    result = model.fit(cov_type="HC2", use_t=True)
    output = {
        "estimate": float(result.params["treatment"]),
        "std_error": float(result.bse["treatment"]),
        "p_value": float(result.pvalues["treatment"]),
    }
    args.output.write_text(json.dumps(output, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
