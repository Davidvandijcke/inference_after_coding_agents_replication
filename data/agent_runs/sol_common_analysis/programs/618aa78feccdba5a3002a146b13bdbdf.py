#!/usr/bin/env python3
"""Estimate the full-sample intention-to-treat effect on employed_m12.

Usage: python score.py --data data.csv --output result.json
Requires numpy, pandas, scipy, and statsmodels.
The p-value is the unadjusted two-sided HC2 t-test with residual degrees
of freedom; this program fits only the specified employment regression.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error("Output must be separate from the input dataset.")

    data = pd.read_csv(args.data, usecols=["site", "treatment", "employed_m12"])
    if data.empty or data.isna().any().any():
        raise ValueError("The analysis requires complete observations.")
    if not data["treatment"].isin([0, 1]).all():
        raise ValueError("Treatment must be coded 0/1.")
    if not data["employed_m12"].isin([0, 1]).all():
        raise ValueError("Employment must be coded 0/1.")

    design = pd.concat(
        [
            pd.Series(1.0, index=data.index, name="intercept"),
            data[["treatment"]].astype(float),
            pd.get_dummies(data["site"], prefix="site", drop_first=True, dtype=float),
        ],
        axis=1,
    )
    if len(data) <= design.shape[1] or np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("The regression requires positive residual degrees of freedom and full rank.")

    result = sm.OLS(data["employed_m12"].astype(float), design).fit(
        cov_type="HC2", use_t=True
    )
    output = {
        "estimate": float(result.params["treatment"]),
        "std_error": float(result.bse["treatment"]),
        "p_value": float(result.pvalues["treatment"]),
    }
    if not all(np.isfinite(value) for value in output.values()):
        raise ValueError("The regression did not produce finite inference results.")
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
