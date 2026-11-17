#!/usr/bin/env python3
"""Estimate the intention-to-treat effect on employment at 12 months.

Usage: python score.py --data data.csv --output result.json
Dependencies: pandas, numpy, statsmodels.

The sole analysis is OLS of employed_m12 on an intercept, treatment,
and site indicators. Inference uses HC2 standard errors and a two-sided
Student t test with residual degrees of freedom, as in CODEBOOK.md.
The estimate and standard error are in probability units, not percentages.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error("The output path must differ from the input data path.")

    data = pd.read_csv(args.data, usecols=["site", "treatment", "employed_m12"])
    if data.empty or data.isna().any().any():
        parser.error("The analysis requires nonempty, complete observations.")
    if set(data["treatment"].unique()) != {0, 1}:
        parser.error("Treatment must contain both 0 and 1, and no other values.")
    if not data["employed_m12"].isin([0, 1]).all():
        parser.error("employed_m12 must contain only 0 and 1.")

    sites = pd.get_dummies(data["site"], prefix="site", drop_first=True, dtype=float)
    design = pd.concat([data[["treatment"]].astype(float), sites], axis=1)
    design = sm.add_constant(design, has_constant="add")
    if len(data) <= design.shape[1] or np.linalg.matrix_rank(design) < design.shape[1]:
        parser.error("The regression requires a full-rank design and residual degrees of freedom.")

    fit = sm.OLS(data["employed_m12"].astype(float), design).fit(
        cov_type="HC2", use_t=True
    )
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
