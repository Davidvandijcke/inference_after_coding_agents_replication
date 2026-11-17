#!/usr/bin/env python3
"""Estimate the intention-to-treat effect on employment at 12 months.

Usage: python score.py --data data.csv --output result.json
Dependencies: numpy, pandas, statsmodels.

The only fitted model is employed_m12 ~ treatment + categorical site.
Output uses outcome units (0.01 means one percentage point), HC2 standard
errors, and an unadjusted two-sided Student t-test with residual degrees
of freedom, following CODEBOOK.md. No outcome selection is performed.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def analyze(data_path):
    data = pd.read_csv(data_path, usecols=["employed_m12", "treatment", "site"])
    if data.empty or data.isna().any().any():
        raise ValueError("The chosen analysis requires complete, nonempty data.")
    for column in ("employed_m12", "treatment"):
        if not data[column].isin([0, 1]).all():
            raise ValueError(f"{column} must contain only 0 and 1.")

    sites = pd.get_dummies(data["site"], prefix="site", drop_first=True,
                           dtype=float)
    design = pd.concat([data[["treatment"]].astype(float), sites], axis=1)
    design = sm.add_constant(design, has_constant="add")
    if (len(data) <= design.shape[1]
            or np.linalg.matrix_rank(design.to_numpy()) < design.shape[1]):
        raise ValueError("The model needs positive residual degrees of freedom "
                         "and identifiable treatment and site effects.")

    model = sm.OLS(data["employed_m12"].astype(float), design).fit(
        cov_type="HC2", use_t=True
    )
    result = {
        "estimate": float(model.params["treatment"]),
        "std_error": float(model.bse["treatment"]),
        "p_value": float(model.pvalues["treatment"]),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("The model did not produce finite inference results.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error("--output must differ from --data")
    result = analyze(args.data)
    args.output.write_text(json.dumps(result, allow_nan=False) + "\n",
                           encoding="utf-8")


if __name__ == "__main__":
    main()
