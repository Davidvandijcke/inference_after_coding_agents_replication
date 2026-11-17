#!/usr/bin/env python3
"""Estimate the assignment effect on employment at 12 months.

Usage: python score.py --data data.csv --output result.json
Requires pandas, numpy, scipy, and statsmodels.
Model: employed_m12 ~ intercept + treatment + categorical site indicators.
Inference: HC2 covariance and two-sided Student t test with n - rank(X) df.
No subgroup, alternative-outcome, or specification searches are performed.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def chosen_analysis(data_path):
    frame = pd.read_csv(data_path, usecols=["site", "treatment", "employed_m12"])
    if frame.empty or frame.isna().any().any():
        raise ValueError("Analysis columns must be nonempty and fully observed.")
    for column in ("treatment", "employed_m12"):
        if not frame[column].isin([0, 1]).all():
            raise ValueError(f"{column} must contain only 0 and 1.")
    if frame["treatment"].nunique() != 2:
        raise ValueError("Both treatment arms are required.")

    sites = pd.get_dummies(frame["site"], prefix="site", drop_first=True,
                           dtype=float)
    design = pd.concat([
        pd.Series(1.0, index=frame.index, name="intercept"),
        frame["treatment"].astype(float), sites,
    ], axis=1)
    rank = np.linalg.matrix_rank(design.to_numpy())
    if rank != design.shape[1] or len(frame) <= rank:
        raise ValueError("The regression requires full rank and positive residual df.")
    fit = sm.OLS(frame["employed_m12"].astype(float), design).fit(
        cov_type="HC2", use_t=True
    )
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("The regression produced nonfinite inference.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = chosen_analysis(args.data)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
