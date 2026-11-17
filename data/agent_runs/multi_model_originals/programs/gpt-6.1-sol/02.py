#!/usr/bin/env python3
"""Run the frozen Pathways to Work analysis, with no outcome/subgroup search.

Chosen estimand: overall intention-to-treat effect on debt_collections_m12.
Model: OLS with intercept, treatment, and categorical site fixed effects.
Inference: HC2 standard error; two-sided Student t test using residual df.

Requires numpy, pandas, and statsmodels (with scipy).
Usage: python score.py --data data.csv --output result.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def analyze(data_path):
    outcome = "debt_collections_m12"
    data = pd.read_csv(data_path, usecols=["treatment", "site", outcome])
    if data.empty or data.isna().any().any():
        raise ValueError("The analysis columns must be nonempty and fully observed.")
    for column in ("treatment", outcome):
        data[column] = pd.to_numeric(data[column], errors="raise")
        if not data[column].isin([0, 1]).all():
            raise ValueError(f"{column} must contain only 0 and 1.")
    if data.treatment.nunique() != 2:
        raise ValueError("Both treatment arms must be present.")

    sites = pd.get_dummies(data.site.astype("category"), prefix="site",
                           drop_first=True, dtype=float)
    design = pd.concat([pd.Series(1.0, index=data.index, name="intercept"),
                        data.treatment.astype(float), sites], axis=1)
    matrix = design.to_numpy()
    if len(data) <= design.shape[1] or np.linalg.matrix_rank(matrix) != design.shape[1]:
        raise ValueError("The regression must have full rank and positive residual degrees of freedom.")
    fit = sm.OLS(data[outcome].astype(float), design).fit(cov_type="HC2", use_t=True)
    result = {"estimate": float(fit.params["treatment"]),
              "std_error": float(fit.bse["treatment"]),
              "p_value": float(fit.pvalues["treatment"])}
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("The regression did not yield finite estimates and inference.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.data.resolve() == args.output.resolve():
        parser.error("--output must differ from --data")
    result = analyze(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
