#!/usr/bin/env python3
"""Estimate the site-adjusted intention-to-treat effect on employment at 12 months.

Usage: python score.py --data data.csv --output result.json
Requires numpy, pandas, and statsmodels (and their dependencies).
The sole model is OLS: employed_m12 ~ treatment + categorical site.
Inference uses HC2 standard errors and a two-sided Student t test with
n - rank(X) degrees of freedom, as specified by CODEBOOK.md.
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

    data = pd.read_csv(args.data, usecols=["employed_m12", "treatment", "site"])
    if data.empty or data.isna().any().any():
        raise ValueError("The analysis requires nonempty, complete observations.")
    if not np.isfinite(data[["employed_m12", "treatment"]].to_numpy(dtype=float)).all():
        raise ValueError("Outcome and treatment must be finite numeric values.")
    if set(data["treatment"].unique()) != {0, 1}:
        raise ValueError("Treatment must contain both 0 and 1, and no other values.")
    if not data["employed_m12"].isin([0, 1]).all():
        raise ValueError("Employment must be coded 0 or 1.")

    sites = pd.get_dummies(data["site"], prefix="site", drop_first=True, dtype=float)
    design = sm.add_constant(pd.concat([data[["treatment"]], sites], axis=1),
                             has_constant="add")
    if len(data) <= design.shape[1] or np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("The model requires full rank and positive residual degrees of freedom.")

    fit = sm.OLS(data["employed_m12"], design, missing="raise").fit(
        cov_type="HC2", use_t=True
    )
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("The model did not produce finite estimates and inference.")
    args.output.write_text(json.dumps(result, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
