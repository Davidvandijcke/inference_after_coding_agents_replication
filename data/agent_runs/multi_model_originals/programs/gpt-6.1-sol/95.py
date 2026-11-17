#!/usr/bin/env python3
"""Run the fixed, selected Pathways to Work analysis.

Usage: python score.py --data data.csv --output result.json
Requires pandas, numpy, and statsmodels.

Chosen estimand: intention-to-treat effect on debt_collections_m12 in the
full sample, in probability units (0.01 = one percentage point).
Model: outcome ~ intercept + treatment + categorical site indicators.
Inference: HC2 robust standard error; two-sided t test using n - rank(X)
residual degrees of freedom. No subgroup selection, covariate search,
outcome selection, or multiple-testing adjustment occurs in this program.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def chosen_analysis(data_path):
    required = ["treatment", "site", "debt_collections_m12"]
    frame = pd.read_csv(data_path, usecols=required)
    if frame.empty or frame.isna().any().any():
        raise ValueError("The analysis columns must be nonempty and fully observed.")
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("Analysis columns must contain finite values.")
    if set(frame.treatment.unique()) != {0, 1}:
        raise ValueError("Both treatment arms, coded 0 and 1, are required.")
    if not frame.debt_collections_m12.isin([0, 1]).all():
        raise ValueError("debt_collections_m12 must be coded 0 or 1.")

    sites = pd.get_dummies(frame.site.astype("category"), prefix="site",
                           drop_first=True, dtype=float)
    design = pd.concat([
        pd.Series(1.0, index=frame.index, name="intercept"),
        frame[["treatment"]].astype(float), sites,
    ], axis=1)
    if (len(frame) <= design.shape[1]
            or np.linalg.matrix_rank(design.to_numpy()) != design.shape[1]):
        raise ValueError("The selected regression requires a full-rank design and residual degrees of freedom.")
    fit = sm.OLS(frame.debt_collections_m12, design).fit(
        cov_type="HC2", use_t=True)
    result = {
        "estimate": float(fit.params["treatment"]),
        "std_error": float(fit.bse["treatment"]),
        "p_value": float(fit.pvalues["treatment"]),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("HC2 inference is undefined for this dataset.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error("The output path must differ from the input data path.")
    result = chosen_analysis(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
